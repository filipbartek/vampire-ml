import functools
import itertools
import logging
import os
import random
import sys
import warnings
from collections import Counter
from collections import defaultdict
from contextlib import suppress
from itertools import count

import clogistic
import cvxpy
import joblib
import hydra
import more_itertools
import numpy as np
import pandas as pd
import scipy
import sklearn
import tensorflow as tf
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

import classifier
import questions
from dense import Dense
from questions.callbacks import TensorBoard
from questions.graphifier import Graphifier
from questions.memory import memory
from questions.solver import Solver
from questions.utils import py_str
from questions.utils import timer
from utils import astype
from utils import save_df
from utils import sparse_equal
from utils import to_absolute_path
from utils import to_tensor
from vampire import szs
from weight import dataset
from weight import empirical
from weight import evaluator
from weight import linear
from weight import proof
from weight.bounded_linear_classifier import BoundedLinearClassifier
from weight.constant_linear_classifier import ConstantLinearClassifier

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
        problem_names = list(itertools.chain.from_iterable(problem_name_lists))
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

        problem_path_datasets = {k: [questions.config.full_problem_path(p) for p in v] for k, v in
                                 problem_name_datasets.items()}

        clausifier = Solver()

        if cfg.workspace_dir is None:
            log.info('Workspace dir not specified. No proofs to load. Quitting.')
            return

        def sample_weight(problem, samples, dist, seed):
            rng = np.random.default_rng(seed)
            signature = clausifier.signature(problem)
            size = (samples, len(cfg.clause_features) + len(signature))
            log.debug(f'{problem}: Sampling {samples} random weight vectors of length {size[1]}.')
            return dist.rvs(size=size, random_state=rng)

        problem_paths_all = list(rng.permutation(sorted(set(itertools.chain.from_iterable(problem_path_datasets.values())))))

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
                                             clause_features=OmegaConf.to_container(cfg.clause_features),
                                             clause_max_len=cfg.clause_max_len)

        lr_kwargs = {'penalty': 'none', 'fit_intercept': False}
        models = {
            #'bounded_1': lambda: BoundedLinearClassifier(clogistic.LogisticRegression(**lr_kwargs), coef_lb=1),
            # 'bounded_0': lambda: BoundedLinearClassifier(clogistic.LogisticRegression(**lr_kwargs), coef_lb=0),
            # 'unbounded': lambda: BoundedLinearClassifier(clogistic.LogisticRegression(**lr_kwargs)),
            # 'unbounded': lambda: BoundedLinearClassifier(clogistic.LogisticRegression(**lr_kwargs)),
            'unbounded': lambda: sklearn.linear_model.LogisticRegression(**lr_kwargs),
            # 'const_1': ConstantLinearClassifier(coef=1)
        }

        def fit_proofs(feature_vectors, proof_searches, seed):
            try:
                ds = dataset.proofs_to_samples(feature_vectors, proof_searches, max_sample_size=cfg.max_sample_size,
                                               rng=np.random.default_rng(seed), join_searches=cfg.join_searches)
            except dataset.NoClausePairsError as e:
                warnings.warn(str(e))
                return None

            def fit_one_model(model):
                result = {}
                with timer() as t_fit:
                    try:
                        model.fit(**ds)
                    except (ValueError, cvxpy.error.SolverError) as e:
                        result['error'] = {'type': type(e).__name__, 'message': str(e)}
                result['time_fit'] = t_fit.elapsed
                if 'error' not in result:
                    result['model'] = model
                    result['score'] = model.score(**ds)
                return result

            result = {
                'features': ds['X'].shape[1],
                'clause_feature_vectors': len(feature_vectors),
                'clause_pairs': ds['X'].shape[0],
                'model': {name: fit_one_model(model()) for name, model in models.items()}
            }

            return result

        def result_to_weight(result):
            if result is None or 'model' not in result:
                return None
            coefs = [model_res['model'].coef_ for model_res in result['model'].values() if 'model' in model_res]
            if len(coefs) == 0:
                return None
            return np.concatenate(coefs)

        def result_to_record(result):
            if result is None:
                return {}
            record = {k: v for k, v in result.items() if k in ['weight_idx', 'out_dir', 'plot_time', 'probe']}
            if 'verbose' in result:
                record['verbose'] = {k: v for k, v in result['verbose'].items() if k != 'clause_feature_vectors'}
                clauses = {role: sum(fv['role_proof'][role] for fv in result['verbose']['clause_feature_vectors'].values()) for role in [False, True]}
                record['verbose']['clauses'] = {
                    'total': sum(clauses.values()),
                    'nonproof': clauses[False],
                    'proof': clauses[True],
                    'unique_feature_vectors': len(result['verbose']['clause_feature_vectors'])
                }
            return record

        problem_common_path = os.path.commonpath(problem_paths_all)

        def loop_on_problem(problem, seed):
            rng = np.random.default_rng(seed)
            feature_vectors = {}
            feature_vector_hashes = []
            proof_searches = []
            proof_feature_vector_hashes = set()
            records = []
            all_weights = []
            problem_name = os.path.relpath(problem, problem_common_path)
            problem_dir = os.path.join('problem', problem_name.replace('/', '_'))
            for i in tqdm(range(cfg.loop_iterations), desc=problem_name):
                iteration_dir = os.path.join(problem_dir, 'iteration', str(i))
                if i == 0:
                    fit_result = None
                    problem_weights = sample_weight(problem, cfg.initial.size, dist, rng)
                else:
                    def process(proof_search):
                        # TODO: Allow copying all positive clauses to all proofs.
                        false_negatives = {k: v for k, v in proof_search[False].items() if k in proof_feature_vector_hashes}
                        if len(false_negatives) == 0:
                            return proof_search
                        result = {k: v.copy() for k, v in proof_search.items()}
                        result[False].subtract(false_negatives)
                        result[True].update(false_negatives)
                        result[False] = Counter({k: v for k, v in result[False].items() if v > 0})
                        return result

                    proof_searches_processed = [process(v) for v in proof_searches]
                    fit_result = fit_proofs(feature_vectors, proof_searches_processed, rng)
                    problem_weights = result_to_weight(fit_result)
                all_weights.append(problem_weights)
                added = False
                if problem_weights is not None:
                    eval_results = eval_empirical.evaluate({problem: problem_weights}, out_dir=iteration_dir,
                                                           iteration=str(i))
                    for j, result in enumerate(eval_results):
                        record = {'iteration': i, **result_to_record(result)}
                        if fit_result is not None:
                            record['fit'] = {k: v for k, v in fit_result.items() if k != 'model'}
                            model_name = list(fit_result['model'].keys())[j]
                            record['fit']['model'] = {
                                'name': model_name,
                                'score': fit_result['model'][model_name]['score']
                            }
                        records.append(record)
                        assert result['problem'] == problem
                        if 'verbose' in result:
                            p = result['verbose']['clause_feature_vectors']
                            if len(p) == 0:
                                warnings.warn('Empty proof search.')
                                continue
                            added = True
                            proof_search = {role_proof: Counter() for role_proof in [False, True]}
                            for feature_vector_hash, v in p.items():
                                assert all(vv >= 0 for vv in v['role_proof'].values())
                                assert any(vv > 0 for vv in v['role_proof'].values())
                                if feature_vector_hash not in feature_vectors:
                                    assert len(feature_vectors) == len(feature_vector_hashes)
                                    feature_vectors[feature_vector_hash] = v['feature_vector']
                                    feature_vector_hashes.append(feature_vector_hash)
                                assert sparse_equal(feature_vectors[feature_vector_hash], v['feature_vector'])
                                feature_vector_index = feature_vector_hashes.index(feature_vector_hash)
                                for role_proof in [False, True]:
                                    if v['role_proof'][role_proof] == 0:
                                        continue
                                    proof_search[role_proof][feature_vector_index] += v['role_proof'][role_proof]
                            proof_searches.append(proof_search)
                            proof_feature_vector_hashes.update(proof_search[True])
                df = pd.json_normalize(records, sep='_')
                df.set_index(['iteration', 'weight_idx'], inplace=True)
                save_df(df, os.path.join(problem_dir, 'iterations'))
                if problem_weights is not None:
                    weight_dict = eval_empirical.weight_vector_to_dict(problem, problem_weights[0])[0]
                    header = ','.join([k for k in weight_dict.keys() if k != 'symbol'] + list(weight_dict['symbol'].keys())[1:])
                    np.savetxt(os.path.join(problem_dir, 'weights.csv'), np.concatenate(all_weights), delimiter=',', header=header, comments='')
                if not added:
                    break
            return records

        results = joblib.Parallel(verbose=cfg.parallel.verbose)(joblib.delayed(loop_on_problem)(p, seed) for p, seed in zip(problem_paths_all, ss.spawn(len(problem_paths_all))))
        records = []
        for problem, iterations in zip(problem_paths_all, results):
            record = {
                'problem': problem,
                'iterations': iterations[-1]['iteration'] + 1,
                'initial': {
                    'total': cfg.initial.size,
                    'solved': sum(szs.is_solved(r['probe']['szs_status']) for r in iterations[:cfg.initial.size]),
                    'uns': sum(szs.is_unsat(r['probe']['szs_status']) for r in iterations[:cfg.initial.size])
                },
                'trained': {
                    'total': len(iterations) - cfg.initial.size,
                    'solved': sum(szs.is_solved(r['probe']['szs_status']) for r in iterations[cfg.initial.size:]),
                    'uns': sum(szs.is_unsat(r['probe']['szs_status']) for r in iterations[cfg.initial.size:])
                },
            }
            records.append(record)
        df = pd.json_normalize(records)
        save_df(df, 'problems')
        return

        proofs = defaultdict(dict)
        problem_weights = random_weights
        for i in range(100):
            out_dir = os.path.join('loop', str(i))
            # eval_plot.evaluate(problem_weights, out_dir=out_dir)
            eval_results = eval_empirical.evaluate(problem_weights, out_dir=out_dir, iteration=str(i),
                                                   parallel=parallel)
            for result in eval_results:
                problem = result['problem']
                if 'verbose' in result:
                    p = result['verbose']['clause_feature_vectors']
                    if len(p) == 0:
                        warnings.warn('Empty proof search.')
                        del proofs[problem]
                        continue
                    for role_proof in [False, True]:
                        n_clauses = sum(v['role_proof'][role_proof] for v in p.values())
                        if n_clauses == 0:
                            continue
                        for feature_vector_hash, v in p.items():
                            if feature_vector_hash not in proofs[problem]:
                                proofs[problem][feature_vector_hash] = {
                                    'feature_vector': v['feature_vector'],
                                    'role_proof': {role_proof: 0 for role_proof in [False, True]}
                                }
                            proofs[problem][feature_vector_hash]['role_proof'][role_proof] += v['role_proof'][
                                                                                                  role_proof] / n_clauses
            results = parallel(
                joblib.delayed(fit_proofs)(p, seed) for p, seed in zip(proofs.values(), ss.spawn(len(proofs))))
            problem_weights = {problem: result_to_weight(result) for problem, result in
                               zip(proofs, results) if result is not None}
            problem_weights = {k: v for k, v in problem_weights.items() if v is not None}
            proofs = {k: v for k, v in proofs.items() if k in problem_weights}

        def analyze(problem, samples_aggregated, seed):
            if samples_aggregated is None:
                return None
            token_counts = samples_aggregated['token_counts']
            p = samples_aggregated['proof']
            rng = np.random.default_rng(seed)
            record = {
                'features': token_counts.shape[1],
                'clauses': {
                    'original': {
                        'total': p.shape[0],
                        'proof': p.nnz,
                        'nonproof': p.shape[0] - p.nnz
                    }
                }
            }
            samples_aggregated['token_counts'], samples_aggregated['proof'] = proof.subsample_proof(token_counts, p,
                                                                                                    cfg.max_sample_size,
                                                                                                    rng)
            p = samples_aggregated['proof']
            record['clauses']['subsampled'] = {
                'total': p.shape[0],
                'proof': p.nnz,
                'nonproof': p.shape[0] - p.nnz
            }
            lr_kwargs = {'penalty': 'none', 'fit_intercept': False, 'max_iter': 1000}
            models = {
                'bounded_1': BoundedLinearClassifier(clogistic.LogisticRegression(**lr_kwargs), coef_lb=1),
                'bounded_0': BoundedLinearClassifier(clogistic.LogisticRegression(**lr_kwargs), coef_lb=0),
                'unbounded': sklearn.linear_model.LogisticRegression(**lr_kwargs),
                'const_1': ConstantLinearClassifier(coef=1)
            }
            for name, model in models.items():
                out_dir = os.path.join('linear', name)

                def evaluate_weights(weights):
                    options = {**cfg.options.common, **cfg.options.probe, **cfg.options.evaluation.default}
                    return empirical.evaluate_one(problem, weights, clausifier, options, cfg, out_dir)

                try:
                    # Raises `ValueError` if the solver gets data with only one class, that is if there is only one positive and one negative clause.
                    record[name] = linear.analyze_pair(model, samples_aggregated, cfg.clause_features,
                                                       evaluate_weights=evaluate_weights)
                except (cvxpy.error.SolverError, ValueError) as e:
                    log.warning(str(e))
                    record[name] = {'error': {'type': type(e).__name__, 'message': str(e)}}
            return record

        print(f'Analyzing {len(proof_traces)} proofs...', file=sys.stderr)
        proof_analyses = parallel(joblib.delayed(analyze)(t['problem'], t.get('clauses'), seed) for t, seed in
                                  zip(proof_traces, ss.spawn(len(proof_traces))))

        proof_records = []
        for proof_path, t, pa in zip(proof_paths, proof_traces, proof_analyses):
            rec = {
                'proof': proof_path,
                **{k: v for k, v in t.items() if k not in ['signature', 'clauses']}
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
                rec['clause_max'] = {k: t['clauses']['token_counts'][:, i].max() for i, k in
                                     enumerate(cfg.clause_features)}
                rec['clause_max']['symbols'] = t['clauses']['token_counts'][:, len(cfg.clause_features):].max()
                rec['clause_features'] = t['clauses']['token_counts'].shape[1]
                rec['symbols_x_clauses'] = rec['symbols'] * rec['clauses']['total']
                rec['clauses_x_clause_features'] = rec['clauses']['total'] * rec['clause_features']
            if pa is not None:
                rec['pa'] = pa
            proof_records.append(rec)
        proof_df = pd.json_normalize(proof_records, sep='_')
        proof_df.set_index('proof', inplace=True)
        cols = ['symbols', 'clause_features', 'symbols_x_clauses', 'clauses_x_clause_features', 'clauses_.*',
                'clause_max_*', 'pa_features', 'pa_clauses_*']
        proof_df = astype(proof_df, {k: pd.UInt64Dtype() for k in cols})
        save_df(proof_df, 'proofs')

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
                cfg.batch.size, cfg.proof_sample_weight, max_sample_size=cfg.max_sample_size,
                rng=np.random.default_rng(ss.spawn(1)[0])).cache() for
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
            with tf.name_scope('loss'):
                clause_pair_loss = -tf.math.log_sigmoid(clause_pair_weights, name='clause_pair')
                problem_loss = tf.reduce_mean(clause_pair_loss, axis=1, name='problem')
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
                'accuracy': tf.math.reduce_mean(tf.cast(clause_pair_weights > 0, tf.float64), 1, name='accuracy')
            }

        def train_step(model, x, y):
            with tf.GradientTape() as tape:
                stats = test_step(model, x, y, training=True)
                problem_loss = stats['loss']
                with tf.name_scope('loss'):
                    # https://stackoverflow.com/a/50165189/4054250
                    problem_loss_without_nans = tf.where(tf.math.is_nan(problem_loss, name='is_nan'),
                                                         tf.zeros_like(problem_loss),
                                                         problem_loss, name='without_nans')
                    loss_value = tf.reduce_sum(problem_loss_without_nans, name='sum')
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

        evaluate_empirical = functools.partial(empirical.evaluate, clausifier=clausifier, cfg=cfg, parallel=parallel,
                                               writers=writers)

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


def dict_to_batches(problems, batch_size, proof_clause_weight=0.5, max_sample_size=None, rng=None):
    # `tf.data.Dataset.batch` cannot batch structured input with variably-shaped entries.

    def gen_samples():
        for problem, data in problems.items():
            if len(data) == 0:
                continue
            token_counts = scipy.sparse.vstack((d['token_counts'] for d in data), format='csr')
            p = scipy.sparse.vstack((d['proof'] for d in data), format='csc')

            if p.nnz == 0 or p.shape[0] - p.nnz == 0:
                log.debug(f'Proof on problem {problem} has purely positive or negative samples.')
                # Since we train on pairs of clauses, we cannot learn anything from this proof.
                continue

            try:
                token_counts, p = proof.subsample_proof(token_counts, p, max_sample_size, rng)
            except ValueError as e:
                log.debug(f'Skipping problem {problem}: {e}')
                continue

            yield {'problem': problem,
                   'occurrence_count': token_counts,
                   'proof': p}

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
