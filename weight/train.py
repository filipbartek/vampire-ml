import glob
import itertools
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
import scipy
import tensorflow as tf

import classifier
import questions
from questions import models
from questions.graphifier import Graphifier
from questions.memory import memory
from questions.solver import Solver
from utils import save_df
from utils import to_tensor
from weight import vampire

log = logging.getLogger(__name__)


def occurrence_count_vector(token_counts, symbol_name_to_index, dtype=np.uint8):
    """
    0. variable
    1. negation
    2. equality
    3. inequality
    4:. symbols
    """
    data = [sum(token_counts['variable']), *(token_counts[k] for k in ['negation', 'equality', 'inequality'])]
    if any(d > 0 for d in data):
        data, indices = map(list, zip(*((d, i) for i, d in enumerate(data) if d != 0)))
    else:
        data, indices = [], []
    data += token_counts['symbol'].values()
    assert all(r <= np.iinfo(dtype).max for r in data)
    indices += [4 + symbol_name_to_index[s] for s in token_counts['symbol'].keys()]
    assert len(data) == len(indices)
    indptr = [0, len(data)]
    result = scipy.sparse.csr_matrix((data, indices, indptr), shape=(1, 4 + len(symbol_name_to_index)), dtype=dtype)
    return result


def row_to_sample(index, row, signature):
    # Raises `RuntimeError` if parsing of the formula fails.
    formula = row.formula
    proof = row.role_proof
    proof_symbol = '-+'[proof]
    formula_description = f'{proof_symbol} {index}: {formula}'
    try:
        # Raises `pyparsing.ParseException` if parsing of the formula fails.
        # Raises `RecursionError` if the formula is too deep.
        token_counts = vampire.clause.token_counts(formula)
    except (pyparsing.ParseException, RecursionError) as e:
        raise RuntimeError(f'{formula_description}. Failed to parse.') from e
    #log.debug(f'{formula_description}. {token_counts}')
    return {
        'token_counts': occurrence_count_vector(token_counts, signature),
        'proof': proof,
        'goal': row.extra_goal
    }


def df_to_samples(df, signature):
    for index, row in df.iterrows():
        try:
            yield row_to_sample(index, row, signature)
        except RuntimeError as e:
            log.debug(str(e))
            continue


def load_proof_samples(stdout_path, signature, max_size=None):
    if max_size is not None:
        cur_size = os.path.getsize(stdout_path)
        if cur_size > max_size:
            raise RuntimeError(f'{stdout_path}: The stdout file is too large: {cur_size} > {max_size}')
    with open(stdout_path) as f:
        stdout = f.read()
    # Raises `ValueError` if no proof is found.
    df = vampire.formulas.extract_df(stdout, roles=['proof', 'active'])
    samples_list = list(df_to_samples(df[df.role_active], signature))
    samples_aggregated = {
        'token_counts': scipy.sparse.vstack(s['token_counts'] for s in samples_list),
        'proof': scipy.sparse.csc_matrix([[s['proof']] for s in samples_list], dtype=bool),
        'goal': scipy.sparse.csc_matrix([[s['goal']] for s in samples_list], dtype=bool)
    }
    return samples_aggregated


@memory.cache
def load_proof(path, signature, max_size=None):
    log.debug(f'Loading proof: {path}')
    with open(os.path.join(path, 'meta.json')) as f:
        meta = json.load(f)
    stdout_path = os.path.join(path, 'stdout.txt')
    symbol_name_to_index = {s: i for i, s in enumerate(signature)}
    try:
        # Raises `RuntimeError` if the output file is too large.
        # Raises `ValueError` if no proof is found in the output file.
        samples = load_proof_samples(stdout_path, symbol_name_to_index, max_size)
    except (RuntimeError, ValueError) as e:
        log.warning(f'{stdout_path}: Failed to load proof samples: {str(e)}')
        samples = None
    return meta['problem'], samples


def path_to_problem(path):
    # Path format: runs/{problem}/{seed}/verbose
    path_seed = os.path.dirname(path)
    path_problem = os.path.dirname(path_seed)
    return os.path.basename(path_problem)


@hydra.main(config_path='.', config_name='config', version_base='1.1')
def main(cfg):
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(cfg.tf.run_eagerly)
    tf.summary.experimental.set_step(0)

    log.info(f'cwd: {os.getcwd()}')
    log.info(f'Workspace: {cfg.workspace_dir}')

    rng = np.random.default_rng(cfg.seed)

    problem_names = pd.read_csv(hydra.utils.to_absolute_path(cfg.problems), names=['problem']).problem
    problem_names = rng.permutation(problem_names)
    if cfg.max_problem_count is not None:
        problem_names = problem_names[:cfg.max_problem_count]

    log.info(f'Number of problems: {len(problem_names)}')

    problem_path_to_name = {questions.config.full_problem_path(name): name for name in problem_names}

    def generate_verbose_paths(problem='*'):
        return glob.glob(os.path.join(cfg.workspace_dir, 'runs', problem, '*', 'verbose'))

    proof_paths = generate_verbose_paths()
    problems_with_proofs = set(path_to_problem(path) for path in proof_paths)

    train_count = int(len(problem_names) * cfg.validation_split)
    problem_name_datasets = {
        'train': problem_names[:train_count],
        'val': problem_names[train_count:]
    }
    log.info('Number of problems: %s' % {k: len(v) for k, v in problem_name_datasets.items()})
    problem_with_proof_name_datasets = {k: [p for p in v if p in problems_with_proofs] for k, v in
                                        problem_name_datasets.items()}
    log.info('Number of problems with proofs: %s' % {k: len(v) for k, v in problem_with_proof_name_datasets.items()})
    problem_name_datasets['train'] = problem_with_proof_name_datasets['train']
    log.info('Number of problems: %s' % {k: len(v) for k, v in problem_name_datasets.items()})
    problem_names = list(itertools.chain.from_iterable(problem_name_datasets.values()))
    log.info(f'Number of problems: {len(problem_names)}')

    with joblib.parallel_backend(cfg.parallel.backend, n_jobs=cfg.parallel.n_jobs):
        parallel = joblib.Parallel(verbose=cfg.parallel.verbose)

        clausifier = Solver()

        def get_signature(problem):
            try:
                # Raises `RuntimeError` when clausification fails
                return clausifier.signature(problem)
            except RuntimeError:
                return None

        print(f'Collecting signatures of {len(problem_names)} problems', file=sys.stderr)
        signatures = parallel(joblib.delayed(get_signature)(problem) for problem in problem_names)
        problem_to_signature = dict(zip(problem_names, signatures))

        all_problem_names = itertools.chain.from_iterable(problem_name_datasets.values())
        problem_signatures = {path: problem_to_signature[path] for path in all_problem_names}

        if cfg.evaluate.baseline:
            evaluate(None, problem_signatures, cfg, parallel, problem_name_datasets, 'baseline')

        def generate_paths(problem_to_signature):
            for problem_name, signature in problem_to_signature.items():
                # We automatically omit problems that do not have any proof.
                for path in generate_verbose_paths(problem_name):
                    yield path, signature

        if cfg.workspace_dir is None:
            raise RuntimeError('Input workspace directory path is required.')
        print(f'Loading proofs of {len(problem_to_signature)} problems', file=sys.stderr)
        proof_traces = parallel(
            joblib.delayed(load_proof)(path, signature, cfg.max_proof_stdout_size) for path, signature in
            generate_paths(problem_to_signature))

        problem_samples = defaultdict(list)
        for path, samples in proof_traces:
            if samples is None:
                continue
            problem_samples[problem_path_to_name[path]].append(samples)

        log.info(f'Number of problems with some samples: {len(problem_samples)}')

        graphifier = Graphifier(clausifier, max_number_of_nodes=10000)
        graphs, graphs_df = graphifier.get_graphs_dict(problem_names)
        log.info(f'Number of graphs: {len(graphs)}')

        output_ntypes = ['predicate', 'function']
        gcn = models.symbol_features.GCN(cfg.gcn, graphifier.canonical_etypes, graphifier.ntype_in_degrees,
                                         graphifier.ntype_feat_sizes, output_ntypes=output_ntypes)
        # Outputs an embedding for each token.
        model_symbol_embedding = models.symbol_features.Graph(graphifier, gcn)
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

        datasets_batched = {
            dataset_name: dict_to_batches(
                {problem_name: problem_samples[problem_name] for problem_name in problem_names},
                cfg.batch.size).cache() for dataset_name, problem_names in
            problem_with_proof_name_datasets.items()}

        ckpt_dir = 'ckpt'
        log.info(f'Checkpoint directory: {ckpt_dir}')
        cbs = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(ckpt_dir, 'epoch', 'weights.{epoch:05d}.tf'),
                save_weights_only=True, verbose=0),
            tf.keras.callbacks.EarlyStopping(**cfg.early_stopping),
            tf.keras.callbacks.ReduceLROnPlateau(**cfg.reduce_lr_on_plateau)
        ]

        if len(problem_with_proof_name_datasets['val']) >= 1:
            cbs.append(tf.keras.callbacks.ModelCheckpoint(
                os.path.join(ckpt_dir, 'acc', 'weights.{epoch:05d}-{val_binary_accuracy:.2f}.tf'),
                save_weights_only=True, verbose=1, monitor='val_binary_accuracy', save_best_only=True))

        def evaluate_all(era_dir):
            log.info(f'Evaluating {era_dir}')
            for dataset_name, dataset in datasets_batched.items():
                log.info(f'{dataset_name}: Evaluating...')
                res = model_logit.evaluate(dataset, return_dict=True)
                log.info(f'{dataset_name}: {res}')
            evaluate(model_symbol_weight, problem_signatures, cfg, parallel, problem_name_datasets,
                     os.path.join(era_dir, 'problems'))

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


def evaluate(model, problem_signatures, cfg, parallel, problem_name_datasets, out_dir):
    model_result = None
    if model is not None:
        log.info('Evaluating a model')
        problem_names = list(map(str, problem_signatures.keys()))
        model_result = model.predict(problem_names)
    else:
        log.info('Evaluating baseline')

    for eval_name, eval_options in cfg.options.evaluation.items():
        log.info(f'Evaluating configuration {eval_name}')
        df = evaluate_options(model_result, problem_signatures, cfg, eval_options, parallel)
        for dataset_name, problem_names in problem_name_datasets.items():
            df[f'dataset_{dataset_name}'] = df.problem.isin(problem_names)
            cur_df = df[df.problem.isin(problem_names)]
            n_success = cur_df.szs_status.isin(['THM', 'CAX', 'UNS', 'SAT', 'CSA']).sum()
            log.info(f'{eval_name} {dataset_name} empirical success count: {n_success}/{len(cur_df)}')
        save_df(df, os.path.join(out_dir, 'problems'))


def evaluate_options(model_result, problem_signatures, cfg, eval_options, parallel):
    problem_names = list(map(str, problem_signatures.keys()))
    options = {**cfg.options.common, **cfg.options.probe, **eval_options}

    def run(problem, valid, cost):
        log.debug(f'Attempting problem {problem}')
        result = {'problem': problem, 'valid': valid}
        if not valid:
            return result
        if cost is not None:
            weight = cost.numpy()
            signature = problem_signatures[problem]
            assert len(weight) == 4 + len(signature)
            weights = {
                'variable': weight[0],
                'negation': weight[1],
                'equality': weight[2],
                'inequality': weight[3],
                'symbol': dict(zip(signature, weight[4:]))
            }
            log.debug(f'{problem} {weights}')
        else:
            weights = None
        # result['weights'] = weights
        problem_path = questions.config.full_problem_path(problem)
        try:
            run_result = vampire_run(problem_path, options, weights, vampire=cfg.vampire_cmd, **cfg.probe_run_args)
        except RuntimeError as e:
            warnings.warn(str(e))
            return None
        selected_properties = ['szs_status', 'terminationreason', 'returncode', 'elapsed', 'out_dir',
                               'stdout_len', 'stderr_len']
        result.update({k: run_result[k] for k in selected_properties if k in run_result})
        return result

    if model_result is None:
        cases = zip(problem_names, itertools.repeat(True), itertools.repeat(None))
    else:
        cases = zip(problem_names, model_result['valid'], model_result['costs'])
    print(f'Running {len(problem_names)} cases', file=sys.stderr)
    results = parallel(joblib.delayed(run)(problem, valid, cost) for problem, valid, cost in cases)

    return pd.json_normalize(results, sep='_')


def vampire_run(problem_path, options, weights, *args, **kwargs):
    options = options.copy()
    if 'include' in options and options['include'] is None:
        del options['include']
    if weights is not None:
        options['variable_weight'] = weights['variable']
        # TODO: Set weights for negation, equality, inequality.
        with tempfile.NamedTemporaryFile('w+', suffix='.properties',
                                         prefix=os.path.join('vampire_functor_weights_')) as f:
            for functor, weight in weights['symbol'].items():
                if functor == '=':
                    continue
                f.write(f'{functor}={weight}\n')
            f.seek(0)
            options['functor_weight'] = f.name
            return vampire.run(problem_path, options, *args, **kwargs)
    else:
        return vampire.run(problem_path, options, *args, **kwargs)


def dict_to_batches(problems, batch_size):
    # `tf.data.Dataset.batch` cannot batch structured input with variably-shaped entries.

    def gen_samples():
        for problem, data in problems.items():
            token_counts = scipy.sparse.vstack(d['token_counts'] for d in data)
            proof = scipy.sparse.vstack(d['proof'] for d in data)
            yield {'problem': problem, 'occurrence_count': token_counts, 'proof': proof}

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
