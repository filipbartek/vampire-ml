import itertools
import logging
import os
import random
import sys

import joblib
import hydra
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import yaml
from omegaconf import OmegaConf

import classifier
import questions
import training
from dense import Dense
from questions import models
from questions.graphifier import Graphifier
from questions.memory import memory
from questions.solver import Solver
from training import InitialDesign
from training import StepTimer
from training import Training
from utils import save_df
from weight import evaluator

log = logging.getLogger(__name__)


def path_to_problem(path):
    # Path format: runs/{problem}/{seed}/verbose
    path_seed = os.path.dirname(path)
    path_problem = os.path.dirname(path_seed)
    return os.path.basename(path_problem)


def random_integers(rng, dtype=np.int64, **kwargs):
    rng = np.random.default_rng(rng)
    # Sample integers uniformly from the whole domain of `dtype`.
    ii = np.iinfo(dtype)
    return rng.integers(low=ii.min, high=ii.max, dtype=dtype, endpoint=True, **kwargs)


@hydra.main(config_path='.', config_name='config', version_base='1.1')
def main(cfg):
    if cfg.recursionlimit is not None:
        sys.setrecursionlimit(cfg.recursionlimit)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    
    with joblib.parallel_backend(cfg.parallel.backend, n_jobs=cfg.parallel.n_jobs), tf.device(cfg.tf.device):
        ss = np.random.SeedSequence(cfg.seed)

        rng_seeds = np.random.default_rng(ss.spawn(1)[0])
        # For an unknown reason, nondeterminism from `random` is introduced somewhere in the process.
        random.seed(random_integers(rng_seeds, dtype=int))
        # Seeding `np.random` is just a precaution.
        np.random.seed(random_integers(rng_seeds, dtype=np.uint32))
        tf.random.set_seed(random_integers(rng_seeds))

        rng = np.random.default_rng(ss.spawn(1)[0])

        tf.config.run_functions_eagerly(cfg.tf.run_eagerly)
        tf.debugging.set_log_device_placement(cfg.tf.log_device_placement)
        tf.summary.experimental.set_step(-1)

        writers = {
            'train': tf.summary.create_file_writer(os.path.join(cfg.tensorboard.log_dir, 'train')),
            'val': tf.summary.create_file_writer(os.path.join(cfg.tensorboard.log_dir, 'validation'))
        }

        log.info(f'Working directory: {os.getcwd()}')
        log.info(f'Cache directory: {memory.location}')

        log.info(f'Recursion limit: {sys.getrecursionlimit()}')

        log.info(f'TensorFlow physical devices: \n{yaml.dump(tf.config.experimental.list_physical_devices())}')

        with writers['train'].as_default():
            tf.summary.text('path/cwd', os.getcwd())
            tf.summary.text('path/cache', memory.location)

        if isinstance(cfg.problem.names, str):
            raise RuntimeError('The option problem.names should be a list, not a string.')
        problem_name_lists = [(questions.config.full_problem_path(p) for p in cfg.problem.names)]
        for list_file in cfg.problem.lists:
            dir = os.path.dirname(list_file)
            list = pd.read_csv(hydra.utils.to_absolute_path(list_file), names=['problem']).problem
            abs_paths = (questions.config.full_problem_path(p, [dir]) for p in list)
            problem_name_lists.append(abs_paths)
        problem_paths = sorted(set(itertools.chain.from_iterable(problem_name_lists)))
        problem_paths = rng.permutation(problem_paths)
        if cfg.max_problem_count is not None:
            problem_paths = problem_paths[:cfg.max_problem_count]

        log.info(f'Number of problems: {len(problem_paths)}')

        clausifier = Solver(options={**cfg.options.common, **cfg.options.clausify}, timeout=cfg.clausify_timeout)

        runner_probe = evaluator.VampireRunner(
            options={**cfg.options.common, **cfg.options.probe, **cfg.options.evaluation.default},
            run_kwargs={**cfg.probe_run_args, 'vampire': cfg.vampire_cmd})
        runner_verbose = evaluator.VampireRunner(
            options={**cfg.options.common, **cfg.options.verbose, **cfg.options.evaluation.default},
            run_kwargs={**cfg.probe_run_args, 'vampire': cfg.vampire_cmd})
        eval_empirical = evaluator.Empirical(runner_probe, runner_verbose=runner_verbose, clausifier=clausifier,
                                             clause_features=OmegaConf.to_container(cfg.clause_features),
                                             szs_status_of_interest=cfg.szs_status_of_interest)

        graphifier = Graphifier(clausifier, max_number_of_nodes=cfg.max_problem_nodes, extra_fieldnames=['training'])
        output_ntypes = ['predicate', 'function', 'variable', 'atom', 'equality']
        # Per-node values:
        # - symbol <- predicate, function
        # Readouts:
        # - variable_occurrence <- variable
        # - variable_count <- variable
        # - literal_positive <- atom, equality
        # - literal_negative <- atom, equality
        # - equality <- equality
        # - inequality <- equality
        # - number <- function
        gcn = models.symbol_features.GCN(cfg.gcn, graphifier.canonical_etypes, graphifier.ntype_in_degrees,
                                         graphifier.ntype_feat_sizes, output_ntypes=output_ntypes)
        # Outputs an embedding for each token.
        model_symbol_embedding = models.symbol_features.Graph(graphifier, gcn)
        embedding_to_weight = {
            name: Dense(1, name=name, activation=cfg.embedding_to_cost.activation,
                        output_bias=cfg.embedding_to_cost.output_bias,
                        kernel_regularizer=tf.keras.regularizers.L1L2(**cfg.embedding_to_cost.regularization)) for
            name in cfg.clause_features + ['symbol']
        }
        model_symbol_weight = models.symbol_cost.Composite(model_symbol_embedding, embedding_to_weight,
                                                           l2=cfg.symbol_cost.l2)
        model_logit = classifier.Classifier(model_symbol_weight)

        Optimizer = {
            'sgd': tf.keras.optimizers.SGD,
            'adam': tf.keras.optimizers.Adam,
            'rmsprop': tf.keras.optimizers.RMSprop
        }[cfg.optimizer]
        # We do not scale the learning rate by batch size because the loss is agregated by summation instead of averaging.
        optimizer = Optimizer(learning_rate=cfg.learning_rate)

        model_logit.compile(optimizer=optimizer)

        graphs, graphs_df = graphifier.get_graphs(problem_paths, get_df=True, return_graphs=False)
        save_df(graphs_df, 'graphs')

        os.makedirs('problems', exist_ok=True)
        np.savetxt(os.path.join('problems', 'all.txt'), problem_paths, fmt='%s')

        problem_paths = graphs_df.index[graphs_df.error.isna()]
        log.info(f'Removing problems with too many nodes:\n%s' % yaml.dump({
            'total': {
                'before': len(graphs_df),
                'after': len(problem_paths),
            },
            'error': graphs_df.error.value_counts(dropna=False).to_dict()
        }))
        
        np.savetxt(os.path.join('problems', 'graphified.txt'), problem_paths, fmt='%s')

        train_count = int(len(problem_paths) * cfg.train_ratio)
        subsets = {
            'train': problem_paths[:train_count],
            'val': problem_paths[train_count:]
        }
        
        for name, pp in subsets.items():
            np.savetxt(os.path.join('problems', f'subset_{name}.txt'), pp, fmt='%s')
        
        initial_design = None
        if cfg.initial_design is not None:
            def get_distribution(cfg):
                return getattr(scipy.stats, cfg.name)(**{k: v for k, v in cfg.items() if k != 'name'})

            dist = get_distribution(cfg.initial_design.distribution)
            log.info(f'Initial weight distribution ({cfg.initial_design.distribution}): mean={dist.mean()}, std={dist.std()}')
            initial_design = InitialDesign(clausifier, len(cfg.clause_features), dist, ss.spawn(1)[0])
        
        data = training.Dataset(problem_paths, ss.spawn(1)[0], subsets=subsets)
        tr = Training(data, model=model_logit, evaluator=eval_empirical, optimizer=optimizer, writers=writers,
                      empirical=StepTimer(**cfg.evaluation.empirical), proxy=StepTimer(**cfg.evaluation.proxy),
                      limits=cfg.limits, join_searches=cfg.join_searches, initial_design=initial_design)
        tr.run()


if __name__ == '__main__':
    with pd.option_context('display.max_columns', None,
                           'display.max_rows', None,
                           'display.width', None,
                           'display.float_format', '{:.2f}'.format):
        main()
