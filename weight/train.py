import glob
import itertools
import json
import logging
import os
import random
import sys
import tempfile
import warnings
from collections import defaultdict
from contextlib import suppress

import joblib
import hydra
import more_itertools
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import yaml
from omegaconf import OmegaConf

import classifier
import questions
from questions import models
from questions.graphifier import Graphifier
from questions.memory import memory
from questions.solver import Solver
from utils import json_dump_default
from utils import save_df
from utils import to_tensor
from weight import proof
from weight import signature
from weight import vampire

log = logging.getLogger(__name__)

yaml.add_representer(np.ndarray, yaml.representer.SafeRepresenter.represent_list)
yaml.add_representer(np.int64, yaml.representer.SafeRepresenter.represent_int)
yaml.add_representer(np.uint64, yaml.representer.SafeRepresenter.represent_int)


def path_to_problem(path):
    # Path format: runs/{problem}/{seed}/verbose
    path_seed = os.path.dirname(path)
    path_problem = os.path.dirname(path_seed)
    return os.path.basename(path_problem)


@hydra.main(config_path='.', config_name='config', version_base='1.1')
def main(cfg):
    with joblib.parallel_backend(cfg.parallel.backend, n_jobs=cfg.parallel.n_jobs), tf.device(cfg.tf.device):
        # For an unknown reason, nondeterminism from `random` is introduced somewhere in the process.
        random.seed(0)
        # Seeding `np.random` is just a precaution.
        np.random.seed(0)
        tf.random.set_seed(0)

        tf.config.run_functions_eagerly(cfg.tf.run_eagerly)
        tf.debugging.set_log_device_placement(cfg.tf.log_device_placement)
        tf.summary.experimental.set_step(0)

        log.info(f'Working directory: {os.getcwd()}')
        log.info(f'Workspace directory: {hydra.utils.to_absolute_path(cfg.workspace_dir)}')
        log.info(f'Cache directory: {memory.location}')

        log.info(f'TensorFlow physical devices: \n{yaml.dump(tf.config.experimental.list_physical_devices())}')

        ss = np.random.SeedSequence(cfg.seed)

        problem_name_lists = [cfg.problem.names]
        if cfg.problem.list_file is not None:
            problem_name_lists.append(pd.read_csv(hydra.utils.to_absolute_path(cfg.problem.list_file), names=['problem']).problem)
        problem_names = list(itertools.chain.from_iterable(problem_name_lists))
        problem_names = np.random.default_rng(ss.spawn(1)[0]).permutation(problem_names)
        if cfg.max_problem_count is not None:
            problem_names = problem_names[:cfg.max_problem_count]

        log.info(f'Number of problems: {len(problem_names)}')

        problem_path_to_name = {questions.config.full_problem_path(name): name for name in problem_names}

        def generate_verbose_paths(problem='*'):
            return glob.glob(
                os.path.join(hydra.utils.to_absolute_path(cfg.workspace_dir), 'runs', problem, '*', 'verbose'))

        train_count = int(len(problem_names) * cfg.train_ratio)
        problem_name_datasets = {
            'train': problem_names[:train_count],
            'val': problem_names[train_count:]
        }
        log.info('Number of problems: %s' % {k: len(v) for k, v in problem_name_datasets.items()})
        parallel = joblib.Parallel(verbose=cfg.parallel.verbose)

        clausifier = Solver()

        active_problem_names = list(itertools.chain.from_iterable(problem_name_datasets.values()))

        if cfg.evaluate.baseline:
            evaluate(None, active_problem_names, clausifier, cfg, parallel, problem_name_datasets, 'baseline')

        def generate_paths(problem_names):
            for problem_name in problem_names:
                # We automatically omit problems that do not have any proof.
                for path in generate_verbose_paths(problem_name):
                    yield path

        proof_traces = proof.load_proofs(list(generate_paths(active_problem_names)), clausifier,
                                         OmegaConf.to_container(cfg.clause_features),
                                         cfg=OmegaConf.to_container(cfg.proof), parallel=parallel, ss=ss.spawn(1)[0])

        problem_samples = defaultdict(list)
        max_counts = {
            'clause': 0,
            'token': 0,
            'clause_token': 0,
            'clause_token_unfiltered_nz': 0,
            'proof': 0,
            'nonproof': 0,
            'proof_nonproof': 0
        }
        for res in proof_traces:
            if 'clauses' not in res:
                continue
            samples = res['clauses']
            problem_samples[problem_path_to_name[res['problem']]].append(samples)
            proof_count = samples['proof'].nnz
            nonproof_count = samples['proof'].shape[0] - proof_count
            clause_count = proof_count + nonproof_count
            token_count = samples['token_counts'].shape[1]
            cur_counts = {
                'clause': clause_count,
                'token': token_count,
                'clause_token': clause_count * token_count,
                'clause_token_unfiltered_nz': samples['token_counts'].nnz,
                'proof': proof_count,
                'nonproof': nonproof_count,
                'proof_nonproof': proof_count * nonproof_count
            }
            max_counts = {k: max(v, cur_counts[k]) for k, v in max_counts.items()}

        log.info(f'Number of problems with some samples: {len(problem_samples)}')
        log.info(f'Max counts:\n{yaml.dump(max_counts)}')

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
        embedding_to_weight = tf.keras.layers.Dense(1, name='embedding_to_weight',
                                                    activation='softplus',
                                                    kernel_regularizer=tf.keras.regularizers.L1L2(
                                                        l1=cfg.embedding_to_cost.l1,
                                                        l2=cfg.embedding_to_cost.l2))
        model_symbol_weight = models.symbol_cost.Composite(model_symbol_embedding, embedding_to_weight,
                                                           l2=cfg.symbol_cost.l2,
                                                           common_clause_features=cfg.clause_features)
        model_logit = classifier.Classifier(model_symbol_weight)

        Optimizer = {
            'sgd': tf.keras.optimizers.SGD,
            'adam': tf.keras.optimizers.Adam,
            'rmsprop': tf.keras.optimizers.RMSprop
        }[cfg.optimizer]
        # https://arxiv.org/pdf/1706.02677.pdf
        # https://arxiv.org/abs/1711.00489
        learning_rate = cfg.learning_rate * cfg.batch.size
        optimizer = Optimizer(learning_rate=learning_rate)

        model_logit.compile(optimizer=optimizer)

        datasets_batched = {
            dataset_name: dict_to_batches(
                {problem_name: problem_samples[problem_name] for problem_name in problem_names if
                 len(problem_samples[problem_name]) >= 1},
                cfg.batch.size, cfg.proof_sample_weight).cache() for
            dataset_name, problem_names in
            problem_name_datasets.items()}

        ckpt_dir = 'ckpt'
        log.info(f'Checkpoint directory: {ckpt_dir}')
        cbs = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(ckpt_dir, 'epoch', 'weights.{epoch:05d}.tf'),
                save_weights_only=True, verbose=0),
            tf.keras.callbacks.EarlyStopping(**cfg.early_stopping),
            tf.keras.callbacks.ReduceLROnPlateau(**cfg.reduce_lr_on_plateau)
        ]

        if len(problem_name_datasets['val']) >= 1:
            cbs.append(tf.keras.callbacks.ModelCheckpoint(
                os.path.join(ckpt_dir, 'acc', 'weights.{epoch:05d}-{val_binary_accuracy:.2f}.tf'),
                save_weights_only=True, verbose=1, monitor='val_binary_accuracy', save_best_only=True))

        def evaluate_all(era_dir):
            log.info(f'Evaluating {era_dir}')
            for dataset_name, dataset in datasets_batched.items():
                log.info(f'{dataset_name}: Evaluating...')
                res = model_logit.evaluate(dataset, return_dict=True)
                log.info(f'{dataset_name}: {res}')
            evaluate(model_symbol_weight, active_problem_names, clausifier, cfg, parallel, problem_name_datasets,
                     os.path.join(era_dir, 'eval'))

        if cfg.evaluate.initial:
            era_dir = os.path.join('era', str(0))
            os.makedirs(era_dir, exist_ok=True)
            evaluate_all(era_dir)
        for era in range(cfg.eras):
            era_dir = os.path.join('era', str(era))
            os.makedirs(era_dir, exist_ok=True)
            model_logit.fit(datasets_batched['train'], validation_data=datasets_batched['val'],
                            initial_epoch=cfg.epochs_per_era * era, epochs=cfg.epochs_per_era * (era + 1),
                            callbacks=cbs + [tf.keras.callbacks.CSVLogger(os.path.join(era_dir, 'epochs.csv'))])
            evaluate_all(era_dir)


def evaluate(model, problem_names, clausifier, cfg, parallel, problem_name_datasets, out_dir):
    model_result = None
    if model is not None:
        log.info('Evaluating a model')
        # We convert problem names to Python strings.
        # They may be input as numpy strings.
        # If a list of numpy strings of length 1 is used, `tf.keras.Model.predict` is confused.
        model_result = model.predict(list(map(str, problem_names)), batch_size=cfg.batch.size)
    else:
        log.info('Evaluating baseline')

    for eval_name, eval_options in cfg.options.evaluation.items():
        log.info(f'Evaluating configuration {eval_name}')
        eval_dir = os.path.join(out_dir, eval_name)
        os.makedirs(eval_dir, exist_ok=True)
        df = evaluate_options(model_result, problem_names, clausifier, cfg, eval_options, parallel, out_dir=eval_dir)
        df['success_uns'] = df.szs_status.isin(['THM', 'CAX', 'UNS'])
        df['success_sat'] = df.szs_status.isin(['SAT', 'CSA'])
        df['success'] = df.success_uns | df.success_sat
        stats = {
            'total': {
                'problems': len(df),
                'successes': df.success.sum()
            }
        }
        assert df.index.name == 'problem'
        for dataset_name, problem_names in problem_name_datasets.items():
            df[f'dataset_{dataset_name}'] = df.index.isin(problem_names)
            cur_df = df[df.index.isin(problem_names)]
            log.info(f'{eval_name} {dataset_name} empirical success count: {cur_df.success.sum()}/{len(cur_df)}')
            stats[dataset_name] = {
                'problems': len(cur_df),
                'successes': cur_df.success.sum()
            }
        with open(os.path.join(eval_dir, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=4, default=json_dump_default)
        save_df(df, os.path.join(eval_dir, 'problems'))


def evaluate_options(model_result, problem_names, clausifier, cfg, eval_options, parallel, out_dir=None):
    options = {**cfg.options.common, **cfg.options.probe, **eval_options}

    def run(problem, valid, cost):
        log.debug(f'Attempting problem {problem}')
        result = {'problem': problem, 'valid': valid}
        if not valid:
            return result
        if cost is not None:
            weight = cost
            signature = clausifier.signature(problem)
            assert len(weight) == len(cfg.clause_features) + len(signature)
            weights = {
                **dict(zip(cfg.clause_features, weight)),
                'symbol': dict(zip(signature, weight[len(cfg.clause_features):]))
            }
            assert len(weights['symbol']) == len(signature)
            log.debug(f'{problem} {weights}')
        else:
            weights = None
        # result['weights'] = weights
        problem_path = questions.config.full_problem_path(problem)
        vampire_out_dir = os.path.join(out_dir, 'problems', problem)
        try:
            run_result = vampire_run(problem_path, options, weights, vampire=cfg.vampire_cmd,
                                     weights_filename=os.path.join(vampire_out_dir, 'functor_weight.properties'),
                                     out_dir=vampire_out_dir, **cfg.probe_run_args)
        except RuntimeError as e:
            warnings.warn(str(e))
            return None
        selected_properties = ['szs_status', 'terminationreason', 'returncode', 'elapsed', 'out_dir',
                               'stdout_len', 'stderr_len']
        result.update({k: run_result[k] for k in selected_properties if k in run_result})
        log.debug(f'Attempt result:\n{yaml.dump(result)}')
        return result

    if model_result is None:
        cases = zip(problem_names, itertools.repeat(True), itertools.repeat(None))
    else:
        cases = zip(problem_names, map(bool, model_result['valid']), map(lambda x: x.numpy(), model_result['costs']))
    print(f'Running {len(problem_names)} cases', file=sys.stderr)
    results = parallel(joblib.delayed(run)(problem, valid, cost) for problem, valid, cost in cases)

    df = pd.json_normalize(results, sep='_')
    df.set_index('problem', inplace=True)
    return df


def vampire_run(problem_path, options, weights, *args, weights_filename=None, **kwargs):
    options = options.copy()
    if 'include' in options and options['include'] is None:
        del options['include']
    weights_file = None
    if weights is not None:
        with suppress(KeyError):
            options['variable_weight'] = weights['variable_occurrence']
        # TODO: Set weights for other clause features.
        if weights_filename is None:
            weights_file = tempfile.NamedTemporaryFile('w+', suffix='.properties',
                                                       prefix=os.path.join('vampire_functor_weights_'))
        else:
            os.makedirs(os.path.dirname(weights_filename), exist_ok=True)
            weights_file = open(weights_filename, 'w+')
        for functor, weight in weights['symbol'].items():
            if functor == '=':
                continue
            weights_file.write(f'{functor}={weight}\n')
        weights_file.seek(0)
        options['functor_weight'] = weights_file.name
    result = vampire.run(problem_path, options, *args, **kwargs)
    if weights_file is not None:
        weights_file.close()
    return result


def dict_to_batches(problems, batch_size, proof_clause_weight=0.5):
    # `tf.data.Dataset.batch` cannot batch structured input with variably-shaped entries.

    def gen_samples():
        for problem, data in problems.items():
            if len(data) == 0:
                continue
            token_counts = scipy.sparse.vstack((d['token_counts'] for d in data), format='csr')
            proof = scipy.sparse.vstack((d['proof'] for d in data), format='csc')

            yield {'problem': problem,
                   'occurrence_count': token_counts,
                   'proof': proof}

    dtypes = {'problem': tf.string, 'occurrence_count': tf.float32, 'nonproof': tf.bool, 'sample_weight': tf.float32}

    def gen():
        for b in more_itertools.chunked(gen_samples(), batch_size):
            x = {
                'problem': to_tensor((row['problem'] for row in b), dtype=dtypes['problem'], name='problem'),
                'occurrence_count': to_tensor((row['occurrence_count'] for row in b), dtype=dtypes['occurrence_count'],
                                              name='occurrence_count')
            }
            y = to_tensor((np.logical_not(np.squeeze(row['proof'].toarray(), axis=1)) for row in b),
                          dtype=dtypes['nonproof'], name='nonproof')

            clause_counts = np.asarray([row['proof'].shape[0] for row in b])
            proof_clause_counts = np.asarray([row['proof'].sum() for row in b])
            nonproof_clause_counts = clause_counts - proof_clause_counts
            assert 0 < proof_clause_weight < 1
            proof_clause_weights = proof_clause_weight / proof_clause_counts
            nonproof_clause_weights = (1 - proof_clause_weight) / nonproof_clause_counts

            flat_values = tf.where(y.flat_values,
                                   x=tf.cast(tf.repeat(nonproof_clause_weights, y.row_lengths()),
                                             dtypes['sample_weight']),
                                   y=tf.cast(tf.repeat(proof_clause_weights, y.row_lengths()), dtypes['sample_weight']))
            sample_weight = tf.RaggedTensor.from_nested_row_splits(flat_values, y.nested_row_splits,
                                                                   name='sample_weight')
            yield x, y, sample_weight

    # The first dimension of the shape is the batch size.
    # We specify None instead of `batch_size` because the last batch may be smaller than `batch_size`.
    output_signature = (
        {
            'problem': tf.TensorSpec(shape=(None,), dtype=dtypes['problem']),
            'occurrence_count': tf.RaggedTensorSpec(shape=(None, None, None), dtype=dtypes['occurrence_count'])
        },
        tf.RaggedTensorSpec(shape=(None, None), dtype=dtypes['nonproof']),
        tf.RaggedTensorSpec(shape=(None, None), dtype=dtypes['sample_weight'])
    )
    return tf.data.Dataset.from_generator(gen, output_signature=output_signature)


if __name__ == '__main__':
    main()
