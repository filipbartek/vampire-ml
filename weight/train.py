import glob
import json
import logging
import os
import sys
import tempfile
import warnings
from collections import defaultdict

import joblib
import hydra
import more_itertools
import numpy as np
import pandas as pd
import pyparsing
import tensorflow as tf
from tqdm import tqdm

import classifier
from questions import models
from questions.graphifier import Graphifier
from questions.solver import Solver
from utils import save_df
from utils import to_tensor
from weight import vampire

log = logging.getLogger(__name__)


@hydra.main(config_path='.', config_name='config', version_base='1.1')
def main(cfg):
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(cfg.tf.run_eagerly)
    tf.summary.experimental.set_step(0)

    log.info(f'cwd: {os.getcwd()}')
    log.info(f'Workspace: {cfg.workspace_dir}')

    rng = np.random.default_rng(cfg.seed)

    with joblib.parallel_backend(cfg.parallel.backend, n_jobs=cfg.parallel.n_jobs):
        parallel = joblib.Parallel(verbose=cfg.parallel.verbose)

        clausifier = Solver()

        # problems = pd.read_csv(hydra.utils.to_absolute_path(cfg.problems), names=['problem']).problem
        if cfg.workspace_dir is None:
            raise RuntimeError('Input workspace directory path is required.')

        def load_proof(verbose_path):
            log.debug(f'Loading proof: {verbose_path}')
            with open(os.path.join(verbose_path, 'meta.json')) as f:
                meta = json.load(f)
            problem = meta['problem']
            with open(os.path.join(verbose_path, 'stdout.txt')) as f:
                stdout = f.read()
            samples = []
            try:
                df = vampire.formulas.extract_df(stdout)
                for index, row in df[df.role_active].iterrows():
                    formula = row.formula
                    proof = row.role_proof
                    proof_symbol = '-+'[proof]
                    try:
                        token_counts = vampire.clause.token_counts(formula)
                    except pyparsing.ParseException as e:
                        warnings.warn(f'{proof_symbol} {index}: {formula}. Failed to parse: {str(e)}')
                        continue
                    log.debug(f'{proof_symbol} {index}: {formula}. {token_counts}')
                    sample = {
                        'formula': formula,
                        'token_counts': token_counts,
                        'proof': proof,
                        'goal': row.extra_goal
                    }
                    samples.append(sample)
            except ValueError as e:
                log.debug(str(e))
            return problem, samples

        verbose_paths = glob.glob(os.path.join(cfg.workspace_dir, 'runs', '*', '*', 'verbose'))
        print(f'Loading {len(verbose_paths)} proofs', file=sys.stderr)
        proof_traces = parallel(joblib.delayed(load_proof)(verbose_path) for verbose_path in verbose_paths)

        problem_samples = defaultdict(list)
        for problem, samples in proof_traces:
            if len(samples) == 0:
                continue
            problem_samples[problem].extend(samples)

        log.info(f'Number of problems: {len(problem_samples)}')

        problems = {}
        for problem, samples in tqdm(problem_samples.items(), unit='problem', desc='Collecting problem signatures'):
            assert len(samples) > 0
            predicates = clausifier.symbols_of_type(problem, 'predicate')
            functions = clausifier.symbols_of_type(problem, 'function')
            assert all(
                set(sample['token_counts']['symbol']) <= set(list(predicates.name) + list(functions.name)) for sample in
                samples)
            problems[problem] = {
                'predicates': predicates,
                'functions': functions,
                'samples': samples
            }

        graphifier = Graphifier(clausifier, max_number_of_nodes=10000)
        graphs, graphs_df = graphifier.get_graphs_dict(problems)
        log.info(f'Number of graphs: {len(graphs)}')

        output_ntypes = ['predicate', 'function']
        gcn = models.symbol_features.GCN(cfg.gcn, graphifier.canonical_etypes, graphifier.ntype_in_degrees,
                                         graphifier.ntype_feat_sizes, output_ntypes=output_ntypes)
        model_symbol_embedding = models.symbol_features.Graph(graphifier, 'predicate', gcn)
        embedding_to_weight = tf.keras.layers.Dense(1, name='embedding_to_weight',
                                                    kernel_regularizer=tf.keras.regularizers.L1L2(
                                                        l1=cfg.embedding_to_cost.l1,
                                                        l2=cfg.embedding_to_cost.l2))
        model_symbol_weight = models.symbol_cost.Composite(model_symbol_embedding, embedding_to_weight,
                                                           l2=cfg.symbol_cost.l2)
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

        problem_names = rng.permutation(list(problems.keys()))
        train_count = int(len(problem_names) * cfg.validation_split)
        problem_name_datasets = {
            'train': problem_names[:train_count],
            'val': problem_names[train_count:]
        }
        datasets_batched = {
            dataset_name: dict_to_batches({problem_name: problems[problem_name] for problem_name in problem_names},
                                          cfg.batch.size).cache() for dataset_name, problem_names in
            problem_name_datasets.items()}

        ckpt_dir = 'ckpt'
        log.info(f'Checkpoint directory: {ckpt_dir}')
        cbs = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(ckpt_dir, 'epoch', 'weights.{epoch:05d}.tf'),
                save_weights_only=True, verbose=0),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(ckpt_dir, 'acc', 'weights.{epoch:05d}-{val_binary_accuracy:.2f}.tf'),
                save_weights_only=True, verbose=1, monitor='val_binary_accuracy', save_best_only=True),
            tf.keras.callbacks.EarlyStopping(**cfg.early_stopping),
            tf.keras.callbacks.ReduceLROnPlateau(**cfg.reduce_lr_on_plateau)
        ]

        def evaluate_all(era_dir):
            for dataset_name, dataset in datasets_batched.items():
                res = model_logit.evaluate(dataset, return_dict=True)
                log.info(f'{dataset_name}: {res}')
            df = evaluate(model_symbol_weight, problems, cfg, parallel)
            for dataset_name, problem_names in problem_name_datasets.items():
                cur_df = df[df.problem.isin(problem_names)]
                n_success = cur_df.szs_status.isin(['THM', 'CAX', 'UNS', 'SAT', 'CSA']).sum()
                log.info(f'{dataset_name} empirical success count: {n_success}/{len(cur_df)}')
            save_df(df, os.path.join(era_dir, 'problems'))

        era_dir = os.path.join('era', str(0))
        evaluate_all(era_dir)
        for era in range(cfg.eras):
            era_dir = os.path.join('era', str(era))
            model_logit.fit(datasets_batched['train'], validation_data=datasets_batched['val'],
                            initial_epoch=cfg.epochs_per_era * era, epochs=cfg.epochs_per_era * (era + 1),
                            callbacks=cbs + [tf.keras.callbacks.CSVLogger(os.path.join(era_dir, 'epochs.csv'))])
            evaluate_all(era_dir)


def evaluate(model, problems, cfg, parallel):
    problem_names = list(problems.keys())
    res = model.predict(problem_names)
    options = {**cfg.options.common, **cfg.options.probe}

    def run(problem, valid, cost):
        log.debug(f'Attempting problem {problem}')
        result = {'problem': problem, 'valid': valid}
        if not valid:
            return result
        weight = cost.numpy()
        symbol_names = list(problems[problem]['predicates'].name) + list(problems[problem]['functions'].name)
        weights = {
            'variable': weight[0],
            'negation': weight[1],
            'equality': weight[2],
            'inequality': weight[3],
            'symbol': dict(zip(symbol_names, weight[4:]))
        }
        log.debug(f'{problem} {weights}')
        # result['weights'] = weights
        run_result = vampire_run(problem, options, weights, vampire=cfg.vampire_cmd, **cfg.probe_run_args)
        selected_properties = ['szs_status', 'terminationreason', 'returncode', 'elapsed', 'out_dir',
                               'stdout_len', 'stderr_len']
        result.update({k: run_result[k] for k in selected_properties if k in run_result})
        return result

    cases = zip(problem_names, res['valid'], res['costs'])
    print(f'Running {len(problem_names)} cases', file=sys.stderr)
    results = parallel(joblib.delayed(run)(problem, valid, cost) for problem, valid, cost in cases)

    return pd.json_normalize(results, sep='_')


def vampire_run(problem_path, options, weights, *args, **kwargs):
    options = options.copy()
    if 'include' in options and options['include'] is None:
        del options['include']
    options['variable_weight'] = weights['variable']
    # TODO: Set weights for negation, equality, inequality.
    with tempfile.NamedTemporaryFile('w+', suffix='.properties', prefix=os.path.join('vampire_functor_weights_')) as f:
        for functor, weight in weights['symbol'].items():
            if functor == '=':
                continue
            f.write(f'{functor}={weight}\n')
        f.seek(0)
        options['functor_weight'] = f.name
        return vampire.run(problem_path, options, *args, **kwargs)


def dict_to_batches(problems, batch_size, row_splits_dtype=tf.dtypes.int64):
    # `tf.data.Dataset.batch` cannot batch structured input with variably-shaped entries.

    def gen_samples():
        for problem, data in problems.items():
            if len(data['samples']) == 0:
                continue

            symbols = list(data['predicates'].name) + list(data['functions'].name)

            def occurrence_count_vector(token_counts):
                """
                0. variable
                1. negation
                2. equality
                3. inequality
                4:. symbols
                """
                result = [sum(token_counts['variable'])] + [token_counts[k] for k in
                                                            ['negation', 'equality', 'inequality']] + [
                             token_counts['symbol'][s] for s in symbols]
                return result

            occurrence_count_vector_length = 4 + len(symbols)

            occurrence_counts = [occurrence_count_vector(sample['token_counts']) for sample in data['samples']]
            assert all(len(oc) == occurrence_count_vector_length for oc in occurrence_counts)
            nonproof = [not sample['proof'] for sample in data['samples']]
            yield {'problem': problem, 'occurrence_count': occurrence_counts, 'nonproof': nonproof}

    dtypes = {'problem': tf.string, 'occurrence_count': tf.float32, 'nonproof': tf.bool, 'sample_weight': tf.float32}

    def gen():
        for b in more_itertools.chunked(gen_samples(), batch_size):
            data = {k: to_tensor((row[k] for row in b), dtype=dtypes[k], name=k, row_splits_dtype=row_splits_dtype,
                                 flatten_ragged=False) for k in b[0].keys()}
            x = {k: data[k] for k in ['problem', 'occurrence_count']}
            y = data['nonproof']
            flat_values = tf.constant(1, dtype=dtypes['sample_weight']) / tf.cast(
                tf.repeat(y.row_lengths(), y.row_lengths()), dtypes['sample_weight'])
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
