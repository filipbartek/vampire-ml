import itertools
import logging
import os
import random
import sys
from contextlib import suppress

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
from training import Training
from utils import to_absolute_path
from weight import evaluator

log = logging.getLogger(__name__)

yaml.add_representer(np.ndarray, yaml.representer.SafeRepresenter.represent_list)
yaml.add_representer(np.int64, yaml.representer.SafeRepresenter.represent_int)
yaml.add_representer(np.uint64, yaml.representer.SafeRepresenter.represent_int)


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
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    with joblib.parallel_backend(cfg.parallel.backend, n_jobs=cfg.parallel.n_jobs), tf.device(cfg.tf.device):
        ss = np.random.SeedSequence(cfg.seed)

        rng_seeds = np.random.default_rng(ss.spawn(1)[0])
        # For an unknown reason, nondeterminism from `random` is introduced somewhere in the process.
        random.seed(random_integers(rng_seeds))
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
        log.info(f'Workspace directory: {to_absolute_path(cfg.workspace_dir)}')
        log.info(f'Cache directory: {memory.location}')

        log.info(f'TensorFlow physical devices: \n{yaml.dump(tf.config.experimental.list_physical_devices())}')

        with writers['train'].as_default():
            tf.summary.text('path/cwd', os.getcwd())
            with suppress(ValueError): tf.summary.text('path/workspace', to_absolute_path(cfg.workspace_dir))
            tf.summary.text('path/cache', memory.location)

        if isinstance(cfg.problem.names, str):
            raise RuntimeError('The option problem.names should be a list, not a string.')
        problem_name_lists = [cfg.problem.names]
        if cfg.problem.list_file is not None:
            problem_name_lists.append(
                pd.read_csv(hydra.utils.to_absolute_path(cfg.problem.list_file), names=['problem']).problem)
        problem_names = sorted(set(itertools.chain.from_iterable(problem_name_lists)))
        problem_names = rng.permutation(problem_names)
        if cfg.max_problem_count is not None:
            problem_names = problem_names[:cfg.max_problem_count]

        log.info(f'Number of problems: {len(problem_names)}')
        with writers['train'].as_default():
            tf.summary.scalar('problems/grand_total', len(problem_names))

        train_count = int(len(problem_names) * cfg.train_ratio)
        problem_name_datasets = {
            'train': problem_names[:train_count],
            'val': problem_names[train_count:]
        }
        log.info('Number of problems: %s' % {k: len(v) for k, v in problem_name_datasets.items()})
        for dataset_name, dataset_problems in problem_name_datasets.items():
            with writers[dataset_name].as_default():
                tf.summary.scalar('problems/total', len(dataset_problems))

        clausifier = Solver(options={**cfg.options.common, **cfg.options.clausify}, timeout=cfg.clausify_timeout)

        if cfg.workspace_dir is None:
            log.info('Workspace dir not specified. No proofs to load. Quitting.')
            return

        def get_distribution(cfg):
            return getattr(scipy.stats, cfg.name)(**{k: v for k, v in cfg.items() if k != 'name'})

        dist = get_distribution(cfg.initial.distribution)
        log.info(f'Random weight distribution ({cfg.initial.distribution}): mean={dist.mean()}, std={dist.std()}')
        runner_probe = evaluator.VampireRunner(
            options={**cfg.options.common, **cfg.options.probe, **cfg.options.evaluation.default},
            run_kwargs={**cfg.probe_run_args, 'vampire': cfg.vampire_cmd})
        runner_verbose = evaluator.VampireRunner(
            options={**cfg.options.common, **cfg.options.verbose, **cfg.options.evaluation.default},
            run_kwargs={**cfg.probe_run_args, 'vampire': cfg.vampire_cmd})
        eval_empirical = evaluator.Empirical(runner_probe, runner_verbose=runner_verbose, clausifier=clausifier,
                                             clause_features=OmegaConf.to_container(cfg.clause_features))

        graphifier = Graphifier(clausifier, max_number_of_nodes=cfg.max_problem_nodes)
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

        problem_paths = [questions.config.full_problem_path(p) for p in problem_names]
        train_count = int(len(problem_paths) * cfg.train_ratio)
        subsets = {
            'train': problem_paths[:train_count],
            'val': problem_paths[train_count:]
        }
        data = training.Dataset(problem_paths, ss.spawn(1)[0], subsets=subsets)
        tr = Training(data, model=model_logit, evaluator=eval_empirical, optimizer=optimizer, writers=writers,
                      steps_per_epoch=cfg.steps_per_epoch, train_batch_size=cfg.train_batch_size,
                      predict_batch_size=cfg.predict_batch_size)
        tr.run()


if __name__ == '__main__':
    with pd.option_context('display.max_columns', sys.maxsize,
                           'display.width', None,
                           'display.float_format', '{:.2f}'.format):
        main()
