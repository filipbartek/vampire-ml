import glob
import itertools
import logging
import os
import random
import sys
import tempfile
import warnings
from collections import defaultdict
from contextlib import suppress
from itertools import count

import joblib
import hydra
import more_itertools
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

import classifier
import questions
from dense import Dense
from questions import models
from questions.callbacks import TensorBoard
from questions.graphifier import Graphifier
from questions.memory import memory
from questions.solver import Solver
from questions.utils import py_str
from utils import astype
from utils import save_df
from utils import subsample
from utils import to_str
from utils import to_tensor
from weight import proof
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


def random_integers(rng, dtype=np.int64, **kwargs):
    # Sample integers uniformly from the whole domain of `dtype`.
    ii = np.iinfo(dtype)
    return rng.integers(low=ii.min, high=ii.max, dtype=dtype, endpoint=True, **kwargs)


@hydra.main(config_path='.', config_name='config', version_base='1.1')
def main(cfg):
    with joblib.parallel_backend(cfg.parallel.backend, n_jobs=cfg.parallel.n_jobs), tf.device(cfg.tf.device):
        ss = np.random.SeedSequence(cfg.seed)

        rng_seeds = np.random.default_rng(ss.spawn(1)[0])
        # For an unknown reason, nondeterminism from `random` is introduced somewhere in the process.
        random.seed(random_integers(rng_seeds))
        # Seeding `np.random` is just a precaution.
        np.random.seed(random_integers(rng_seeds, dtype=np.uint32))
        tf.random.set_seed(random_integers(rng_seeds))

        tf.config.run_functions_eagerly(cfg.tf.run_eagerly)
        tf.debugging.set_log_device_placement(cfg.tf.log_device_placement)
        tf.summary.experimental.set_step(-1)

        writers = {
            'train': tf.summary.create_file_writer(os.path.join(cfg.tensorboard.log_dir, 'train')),
            'val': tf.summary.create_file_writer(os.path.join(cfg.tensorboard.log_dir, 'validation'))
        }

        log.info(f'Working directory: {os.getcwd()}')
        log.info(f'Workspace directory: {hydra.utils.to_absolute_path(cfg.workspace_dir)}')
        log.info(f'Cache directory: {memory.location}')

        log.info(f'TensorFlow physical devices: \n{yaml.dump(tf.config.experimental.list_physical_devices())}')

        with writers['train'].as_default():
            tf.summary.text('path/cwd', os.getcwd())
            tf.summary.text('path/workspace', hydra.utils.to_absolute_path(cfg.workspace_dir))
            tf.summary.text('path/cache', memory.location)

        if isinstance(cfg.problem.names, str):
            raise RuntimeError('The option problem.names should be a list, not a string.')
        problem_name_lists = [cfg.problem.names]
        if cfg.problem.list_file is not None:
            problem_name_lists.append(
                pd.read_csv(hydra.utils.to_absolute_path(cfg.problem.list_file), names=['problem']).problem)
        problem_names = list(itertools.chain.from_iterable(problem_name_lists))
        problem_names = np.random.default_rng(ss.spawn(1)[0]).permutation(problem_names)
        if cfg.max_problem_count is not None:
            problem_names = problem_names[:cfg.max_problem_count]

        log.info(f'Number of problems: {len(problem_names)}')
        with writers['train'].as_default():
            tf.summary.scalar('problems/grand_total', len(problem_names))

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
        for dataset_name, dataset_problems in problem_name_datasets.items():
            with writers[dataset_name].as_default():
                tf.summary.scalar('problems/total', len(dataset_problems))

        parallel = joblib.Parallel(verbose=cfg.parallel.verbose)

        clausifier = Solver()

        active_problem_names = list(itertools.chain.from_iterable(problem_name_datasets.values()))

        eval_problem_names = []
        # We spawn a fresh RNG to ensure that changing the number of datasets does not affect subsequent samplings.
        rng_subsamples = np.random.default_rng(ss.spawn(1)[0])
        for k, v in problem_name_datasets.items():
            eval_problem_names.extend(subsample(v, cfg.evaluation_problems[k], rng_subsamples))
        eval_problem_names = sorted(eval_problem_names)

        def generate_paths(problem_names):
            for problem_name in problem_names:
                # We automatically omit problems that do not have any proof.
                for path in generate_verbose_paths(problem_name):
                    yield path

        # We sort the problem names because `load_proofs` is cached.
        proof_paths = list(generate_paths(sorted(active_problem_names)))
        proof_traces = proof.load_proofs(proof_paths, clausifier,
                                         OmegaConf.to_container(cfg.clause_features),
                                         cfg=OmegaConf.to_container(cfg.proof), parallel=parallel, ss=ss.spawn(1)[0])

        proof_records = []
        for proof_path, t in zip(proof_paths, proof_traces):
            rec = {
                'proof': proof_path,
                'problem': t['problem']
            }
            if 'signature' in t:
                rec['symbols'] = len(t['signature'])
            if 'clauses' in t:
                rec['clauses'] = {
                    'total': t['clauses']['token_counts'].shape[0],
                    'proof': t['clauses']['proof'].nnz,
                    'nonproof': t['clauses']['token_counts'].shape[0] - t['clauses']['proof'].nnz,
                    'goal': t['clauses']['goal'].nnz
                }
                rec['clauses']['proof_x_nonproof'] = rec['clauses']['proof'] * rec['clauses']['nonproof']
                rec['max'] = {k: t['clauses']['token_counts'][:, i].max() for i, k in enumerate(cfg.clause_features)}
                rec['clause_features'] = t['clauses']['token_counts'].shape[1]
                rec['symbols_x_clauses'] = rec['symbols'] * rec['clauses']['total']
                rec['clauses_x_clause_features'] = rec['clauses']['total'] * rec['clause_features']
            proof_records.append(rec)
        proof_df = pd.json_normalize(proof_records, sep='_')
        proof_df.set_index('proof', inplace=True)
        proof_df = astype(proof_df, {k: pd.UInt32Dtype() for k in ['symbols', 'clauses_.*', 'max_.*']})
        save_df(proof_df, 'proofs')
        print(proof_df.describe(include='all'))

        problem_samples = defaultdict(list)
        for res in proof_traces:
            if 'clauses' not in res:
                continue
            samples = res['clauses']
            problem_samples[problem_path_to_name[res['problem']]].append(samples)

        log.info(f'Number of problems with some samples: {len(problem_samples)}')
        for dataset_name, dataset_problems in problem_name_datasets.items():
            with writers[dataset_name].as_default():
                tf.summary.scalar('problems/with_proof', len(set(dataset_problems) & set(problem_samples)))

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

        problems_with_proofs = {dataset_name: [n for n in problem_names if n in problem_samples] for
                                dataset_name, problem_names in problem_name_datasets.items()}
        log.info('Number of problems with proofs: %s' % {k: len(v) for k, v in problems_with_proofs.items()})

        Optimizer = {
            'sgd': tf.keras.optimizers.SGD,
            'adam': tf.keras.optimizers.Adam,
            'rmsprop': tf.keras.optimizers.RMSprop
        }[cfg.optimizer]
        # We do not scale the learning rate by batch size because the loss is agregated by summation instead of averaging.
        optimizer = Optimizer(learning_rate=cfg.learning_rate)

        model_logit.compile(optimizer=optimizer)

        datasets_batched = {
            dataset_name: dict_to_batches(
                {problem_name: problem_samples[problem_name] for problem_name in problem_names},
                cfg.batch.size, cfg.proof_sample_weight).cache() for
            dataset_name, problem_names in
            problems_with_proofs.items()}

        ckpt_dir = 'ckpt'
        log.info(f'Checkpoint directory: {ckpt_dir}')

        tensorboard = TensorBoard(**cfg.tensorboard)
        cbs = [
            tensorboard,
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

        def range_count(stop, *args, **kwargs):
            if stop is None:
                return count(*args, **kwargs)
            return range(stop, *args, **kwargs)

        def minimize(optimizer, loss, var_list, tape):
            grads_and_vars = optimizer._compute_gradients(loss, var_list=var_list, tape=tape)
            return optimizer.apply_gradients((grad, var) for grad, var in grads_and_vars if grad is not None)

        def test_step(model, x, y, training=False):
            clause_weights, problem_valid = model(x, training=training)
            # TODO: Allow two losses: 1. clause classifier, 2. clause pair classifier
            clause_pair_weights = model.clause_pair_weights(clause_weights, problem_valid, y)
            clause_pair_loss = -tf.math.log_sigmoid(clause_pair_weights)
            problem_loss = tf.reduce_mean(clause_pair_loss, axis=1)
            if cfg.per_problem_stats:
                for p, cw, cpw, cpl, pl, nonproof in zip(x['problem'], clause_weights, clause_pair_weights,
                                                         clause_pair_loss, problem_loss, y):
                    summary_prefix = f'problem_{py_str(p)}'
                    tf.summary.histogram(f'{summary_prefix}/clause_weight/all', cw)
                    tf.summary.histogram(f'{summary_prefix}/clause_weight/proof', cw[~nonproof])
                    tf.summary.histogram(f'{summary_prefix}/clause_weight/nonproof', cw[nonproof])
                    tf.summary.histogram(f'{summary_prefix}/clause_pair/weight', cpw)
                    tf.summary.histogram(f'{summary_prefix}/clause_pair/loss', cpl)
                    tf.summary.scalar(f'{summary_prefix}/train_loss', pl)
            return {
                'problem': x['problem'],
                'loss': problem_loss,
                'accuracy': tf.math.reduce_mean(tf.cast(clause_pair_weights > 0, tf.float64), 1)
            }

        def train_step(model, x, y):
            with tf.GradientTape() as tape:
                stats = test_step(model, x, y, training=True)
                problem_loss = stats['loss']
                # https://stackoverflow.com/a/50165189/4054250
                problem_loss_without_nans = tf.where(tf.math.is_nan(problem_loss), tf.zeros_like(problem_loss),
                                                     problem_loss)
                loss_value = tf.reduce_sum(problem_loss_without_nans)
            minimize(optimizer, loss_value, model.trainable_weights, tape)
            return stats

        def evaluate_proxy_one(model, dataset, step_fn):
            batch_values = defaultdict(list)
            for step, (x, y, clause_sample_weight) in enumerate(dataset):
                stats = step_fn(model, x, y)
                for k, v in stats.items():
                    batch_values[k].append(v)
            if len(batch_values) == 0:
                return None
            data = {}
            for k, v in batch_values.items():
                values = tf.concat(v, 0)
                if k == 'problem':
                    data[k] = list(map(py_str, values))
                    continue
                data[k] = values
                tf.summary.scalar(f'problems/{k}/total', len(values))
                tf.summary.scalar(f'problems/{k}/invalid', tf.math.count_nonzero(tf.math.is_nan(values)))
                values_without_nans = values[~tf.math.is_nan(values)]
                tf.summary.scalar(f'problems/{k}/valid', len(values_without_nans))
                tf.summary.histogram(f'proxy/{k}', values_without_nans)
                tf.summary.scalar(f'proxy/{k}/mean', tf.reduce_mean(values_without_nans))
                if k == 'loss':
                    tf.summary.scalar(f'proxy/{k}/sum', tf.reduce_sum(values_without_nans))
            df = pd.DataFrame(data=data)
            df.set_index('problem', inplace=True)
            return df

        def evaluate_proxy(model, datasets, step_fn):
            res = {}
            for dataset_name, dataset in datasets.items():
                log.debug(f'Proxy evaluation of dataset {dataset_name}...')
                with writers[dataset_name].as_default():
                    res[dataset_name] = evaluate_proxy_one(model, dataset, step_fn)
            return res

        def run_empirical(model, problem_names, eval_dir):
            log.info('Empirical evaluation...')

            model_result = None
            if model is not None and len(problem_names) > 0:
                log.info('Evaluating a model')
                # We convert problem names to Python strings.
                # They may be input as numpy strings.
                # If a list of numpy strings of length 1 is used, `tf.keras.Model.predict` is confused.
                model_result = model.predict(list(map(str, problem_names)), batch_size=cfg.batch.size)
            else:
                log.info('Evaluating baseline')

            return evaluate_options(model_result, problem_names, clausifier, cfg, cfg.options.evaluation.default,
                                    parallel, out_dir=eval_dir)

        def evaluate_empirical(model, problem_names, problem_name_datasets, eval_dir, summary_prefix='empirical'):
            df = run_empirical(model, problem_names, eval_dir)

            for dataset_name, pn in problem_name_datasets.items():
                with writers[dataset_name].as_default():
                    df[f'dataset_{dataset_name}'] = df.index.isin(pn)
                    cur_df = df[df.index.isin(pn)]
                    tf.summary.scalar(f'{summary_prefix}/problems/total', len(cur_df))
                    if len(cur_df) == 0:
                        continue
                    tf.summary.scalar(f'{summary_prefix}/problems/success', cur_df.success.sum())
                    tf.summary.scalar(f'{summary_prefix}/problems/success_uns', cur_df.success_uns.sum())
                    tf.summary.scalar(f'{summary_prefix}/problems/success_sat', cur_df.success_sat.sum())
                    tf.summary.scalar(f'{summary_prefix}/success_rate', cur_df.success.mean())
                    for col in ['elapsed', 'megainstructions', 'activations']:
                        if col in cur_df:
                            data = cur_df[col][cur_df.success & cur_df[col].notna()]
                            tf.summary.histogram(f'{summary_prefix}/{col}', data)
                    for feature in cfg.clause_features:
                        with suppress(KeyError):
                            data = cur_df[f'weight_{feature}'][cur_df[f'weight_{feature}'].notna()]
                            tf.summary.histogram(f'{summary_prefix}/feature_weight/{feature}', data)
            return df

        baseline_df = None
        if cfg.eval.baseline:
            eval_dir = 'baseline'
            baseline_df = evaluate_empirical(None, eval_problem_names, problem_name_datasets, eval_dir,
                                             summary_prefix='baseline')
            save_df(baseline_df, os.path.join(eval_dir, 'problems'))
        elif cfg.baseline_files is not None:
            baseline_dfs = {name: pd.read_pickle(hydra.utils.to_absolute_path(path)) for name, path in
                            cfg.baseline_files.items()}
            with suppress(KeyError):
                baseline_df = baseline_dfs['default']

        start = 0
        if cfg.eval.initial:
            start = -1
        with tqdm(range_count(cfg.epochs, start=start), unit='epoch', desc='Training') as t, writers[
            'train'].as_default():
            for epoch in t:
                tf.summary.experimental.set_step(epoch)
                if epoch >= 0:
                    train_df = evaluate_proxy_one(model_logit, datasets_batched['train'], train_step)
                    t.set_postfix({col: train_df[col].mean() for col in train_df})
                proxy_results = evaluate_proxy(model_logit, datasets_batched, test_step)
                if cfg.empirical.start is not None and cfg.empirical.step is not None:
                    epoch_rel = epoch - cfg.empirical.start
                    if epoch_rel >= 0 and epoch_rel % cfg.empirical.step == 0:
                        print(f'Empirical evaluation after epoch {epoch}...')
                        # TODO: Save checkpoint.
                        eval_dir = os.path.join('epoch', str(epoch), 'eval')
                        df = evaluate_empirical(model_logit.symbol_weight_model, eval_problem_names,
                                                problem_name_datasets, eval_dir)
                        # Note: Some of `res.values()` may be `None`. `pd.concat` ignores such concatenands.
                        df = df.join(pd.concat(proxy_results.values()))
                        if baseline_df is not None:
                            df = df.join(baseline_df, rsuffix='_baseline')
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
            weights_common = dict(zip(cfg.clause_features, weight))
            weights = {
                **weights_common,
                'symbol': dict(zip(signature, weight[len(cfg.clause_features):]))
            }
            assert len(weights['symbol']) == len(signature)
            if cfg.per_problem_stats:
                result['weight'] = weights
            else:
                result['weight'] = weights_common
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
                               'stdout_len', 'stderr_len', 'megainstructions', 'activations']
        result.update({k: run_result[k] for k in selected_properties if k in run_result})
        log.debug(f'Attempt result:\n{yaml.dump(result)}')
        return result

    if model_result is None:
        cases = zip(problem_names, itertools.repeat(True), itertools.repeat(None))
    else:
        cases = zip(problem_names, map(bool, model_result['valid']), map(lambda x: x.numpy(), model_result['costs']))
    print(f'Running {len(problem_names)} cases', file=sys.stderr)
    results = parallel(joblib.delayed(run)(problem, valid, cost) for problem, valid, cost in cases)

    if cfg.per_problem_stats:
        for result in results:
            if 'weight' not in result:
                continue
            problem = result['problem']
            summary_prefix = f'problem_{problem}/feature_weight'
            for k, v in result['weight']['symbol'].items():
                if k == '=':
                    continue
                tf.summary.scalar(f'{summary_prefix}/symbol/{k}', v)
            del result['weight']['symbol']
            for k, v in result['weight'].items():
                tf.summary.scalar(f'{summary_prefix}/common/{k}', v)

    df = pd.json_normalize(results, sep='_')
    df.set_index('problem', inplace=True)
    df['success_uns'] = df.szs_status.isin(['THM', 'CAX', 'UNS'])
    df['success_sat'] = df.szs_status.isin(['SAT', 'CSA'])
    df['success'] = df.success_uns | df.success_sat
    return df


def vampire_run(problem_path, options, weights, *args, weights_filename=None, **kwargs):
    options = options.copy()
    if 'include' in options and options['include'] is None:
        del options['include']
    weights_file = None
    if weights is not None:
        # TODO: Set weights for other clause features.
        weight_name_to_option_name = {
            'variable_occurrence': 'variable_weight',
            'equality': 'equality_weight'
        }
        options.update({weight_name_to_option_name[weight_name]: v for weight_name, v in weights.items() if
                        weight_name != 'symbol'})
        if weights_filename is None:
            weights_file = tempfile.NamedTemporaryFile('w+', suffix='.txt',
                                                       prefix=os.path.join('vampire_functor_weights_'))
        else:
            os.makedirs(os.path.dirname(weights_filename), exist_ok=True)
            weights_file = open(weights_filename, 'w+')
        for functor, weight in weights['symbol'].items():
            if functor == '=':
                # The weight of equality is passed using a dedicated option.
                # This simplifies the integration within Vampire,
                # since the equality symbol is instantiated before the problem is loaded.
                continue
            weights_file.write(f'{functor} {to_str(weight)}\n')
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
    with pd.option_context('display.max_columns', sys.maxsize,
                           'display.width', None,
                           'display.float_format', '{:.2f}'.format):
        main()
