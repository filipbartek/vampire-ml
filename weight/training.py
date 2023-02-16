import logging
import os
import warnings
from collections import Counter
from collections import defaultdict
from contextlib import suppress

import humanize
import joblib
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
import yaml
from tqdm import tqdm

from questions.utils import flatten_dict
from questions.utils import py_str
from questions.utils import timer
from utils import astype
from utils import get_verbose
from utils import save_df
from utils import sparse_equal
from utils import to_tensor
from vampire import szs
from weight import dataset

log = logging.getLogger(__name__)


class StepTimer:
    def __init__(self, start=-1, step=1):
        self.start = start
        self.step = step

    def is_triggered(self, step):
        step_rel = step - self.start
        return step_rel >= 0 and step_rel % self.step == 0


class Training:
    def __init__(self, dataset, model, optimizer=None, initial_design=None, evaluator=None, writers=None, epochs=None,
                 empirical=None, proxy=None, limits=None, join_searches=False):
        self.data = dataset
        self.model = model
        self.optimizer = optimizer
        self.initial_design = initial_design
        self.evaluator = evaluator
        self.writers = writers
        self.epochs = epochs
        self.empirical = empirical or StepTimer()
        self.proxy = proxy or StepTimer()
        self.limits = limits
        self.join_searches = join_searches
        self.stats = {}
        self.t = None

    def run(self):
        with self.writers['train'].as_default():
            tf.summary.experimental.set_step(-1)
            batches = self.data.generate_batches(subset='train', nonempty=True, join_searches=self.join_searches,
                                                 **self.limits.train)
            stats = {}
            with tqdm(batches, unit='step', desc='Training', disable=not get_verbose(), postfix=stats) as self.t:
                with timer() as time:
                    self.evaluate(initial=True, step=-1)
                self.update_stats({'time/eval': time.elapsed})
                self.summary_cheap()
                for step, batch in enumerate(self.t):
                    with timer() as time_step:
                        tf.summary.experimental.set_step(step)
                        self.update_stats({'batch': {
                            'samples': {
                                'total': len(batch),
                                'subsampled': sum('sample_weight' not in s['clause_pairs'] for s in batch)
                            },
                            'size': {
                                'total': sum(np.prod(s['clause_pairs']['X'].shape) for s in batch),
                                'nnz': sum(s['clause_pairs']['X'].nnz for s in batch)
                            }
                        }})
                        if len(batch) == 0:
                            warnings.warn('Empty batch generated for training.')
                        with timer() as time:
                            self.train_step(batch)
                        self.update_stats({'time/train': time.elapsed})
                        with timer() as time:
                            self.evaluate(step=step)
                        self.update_stats({'time/eval': time.elapsed})
                        self.summary_cheap()
                    self.update_stats({'time/step': time_step.elapsed})
    
    def update_stats(self, stats, subset=None):
        writer = subset
        if writer is None:
            writer = 'train'
        with self.writers[writer].as_default():
            for k, v in flatten_dict(stats, sep='/').items():
                tf.summary.scalar(k, v)
        global_stats = stats
        if subset is not None:
            global_stats = {subset: stats}
        self.stats.update(flatten_dict(global_stats, sep='/'))
        self.t.set_postfix(self.progress_stats())
    
    def progress_stats(self):
        result = {}
        for subset in ['train', 'val']:
            try:
                solved_rate = self.stats[f'{subset}/eval/problems/solved'] / self.stats[f'{subset}/eval/problems/total']
            except KeyError:
                solved_rate = None
            except ZeroDivisionError:
                solved_rate = np.nan
            result.update({
                f'eval/{subset}/solved_rate': solved_rate,
                f'eval/{subset}/acc': self.stats.get(f'{subset}/eval/accuracy'),
                f'eval/{subset}/loss': self.stats.get(f'{subset}/eval/loss')
            })
        result.update({
            'train/acc': self.stats.get('train/train/accuracy'),
            'train/loss': self.stats.get('train/train/loss/mean')
        })
        for k in ['train/acc', 'train/loss']:
            if result[k] is not None:
                result[k] = result[k].numpy()
        return result
                    
    
    def summary_cheap(self):
        memory_info = psutil.Process().memory_full_info()
        log.debug(f'Memory usage:\n{yaml.dump({k: humanize.naturalsize(v, binary=True) for k, v in memory_info._asdict().items()})}')
        fields = ['rss', 'vms', 'shared', 'uss', 'pss', 'swap']
        self.update_stats({'memory': {field: getattr(memory_info, field) for field in fields}})

    def evaluate(self, problems=None, initial=False, step=None):
        eval_empirical = self.empirical.is_triggered(step)
        eval_proxy = self.proxy.is_triggered(step)
        if not (eval_empirical or eval_proxy):
            return None
        if problems is None:
            problems = self.problems()
        dfs = [self.data.problem_stats(problems)]
        if eval_proxy:
            with timer() as time:
                df_problem, df_proof_search = self.evaluate_proxy(problems)
            self.update_stats({'time/eval/proxy': time.elapsed})
            if step is not None:
                save_df(df_proof_search, os.path.join('evaluation', 'proof_search', str(step)))
            with suppress(KeyError):
                dfs.append(df_proof_search[['loss', 'accuracy']].groupby('problem').mean())
        if eval_empirical:
            with timer() as time:
                dfs.append(self.evaluate_empirical(problems, initial=initial, step=step))
            self.update_stats({'time/eval/empirical': time.elapsed})
            if eval_proxy:
                with timer() as time:
                    df_problem, df_proof_search = self.evaluate_proxy(problems)
                self.update_stats({'time/eval/proxy_after': time.elapsed})
                if step is not None:
                    save_df(df_proof_search, os.path.join('evaluation', 'proof_search_after_empirical', str(step)))
                dfs.append(df_proof_search[['loss', 'accuracy']].groupby('problem').mean().add_prefix('after_'))
        df = pd.concat(dfs, axis='columns', copy=False)
        for col in ['feature_weight_predicted', 'probe_solved', 'probe_unsat', 'verbose_solved', 'verbose_unsat']:
            if col in df:
                df[col].fillna(False, inplace=True)
        if step is not None:
            save_df(df, os.path.join('evaluation', 'problem', str(step)))
        for subset, problems in self.data.subsets.items():
            cur_df = df[df.index.isin(problems)]
            with self.writers[subset].as_default():
                problem_stats = {
                    'total': len(cur_df),
                    'with_proof_clauses': (cur_df.proof_feature_vectors > 0).sum(),
                    'with_clauses': (cur_df.feature_vectors > 0).sum(),
                }
                if eval_empirical:
                    problem_stats['feature_weight_predicted'] = cur_df.feature_weight_predicted.sum()
                    problem_stats['solved'] = cur_df.probe_solved.sum()
                    problem_stats['unsat'] = cur_df.probe_unsat.sum()
                record = {k: cur_df[k].mean() for k in ['loss', 'accuracy', 'after_loss', 'after_accuracy'] if k in cur_df}
                record['problems'] = problem_stats
                self.update_stats({'eval': record}, subset=subset)

    def evaluate_empirical(self, problems=None, initial=False, step=None):
        if initial and self.initial_design is not None:
            problem_weights = self.initial_design.sample_weights(problems)
        else:
            print(f'Empirical evaluation: Predicting clause feature weights for {len(problems)} problems...')
            model_result = self.model.symbol_weight_model.predict(problems,
                                                                  batch_size=self.limits.predict.max_batch_samples)
            # Contains two entries: valid, costs
            problem_weights = {}
            for problem, cost, valid in zip(problems, model_result['costs'], model_result['valid']):
                assert problem not in problem_weights
                if valid:
                    problem_weights[problem] = cost
                else:
                    problem_weights[problem] = None
        out_dir = os.path.join('evaluation', 'empirical', str(step))
        empirical_results = self.evaluator.evaluate(problem_weights.items(), out_dir=out_dir)
        if step is not None and step < 0:
            self.data.update(empirical_results, step=step)

        def empirical_result_to_record(empirical_result):
            def run_result_to_record(run_result):
                if run_result is None:
                    return None
                res = {k: v for k, v in run_result.items() if k != 'clause_feature_vectors'}
                res['solved'] = szs.is_solved(run_result['szs_status'])
                res['unsat'] = szs.is_unsat(run_result['szs_status'])
                with suppress(KeyError):
                    feature_vectors = run_result['clause_feature_vectors']
                    res['unique_feature_vectors'] = len(feature_vectors)
                    res['clauses'] = {
                        'proof': sum(fv['role_proof'][True] for fv in feature_vectors.values()),
                        'nonproof': sum(fv['role_proof'][False] for fv in feature_vectors.values())
                    }
                return res

            res = {
                'problem': empirical_result['problem'],
                **{k: run_result_to_record(empirical_result[k]) for k in ['probe', 'verbose'] if k in empirical_result}
            }

            weight = problem_weights[empirical_result['problem']]
            res['feature_weight_predicted'] = weight is not None
            if weight is not None:
                weight_static = weight[:len(self.evaluator.clause_features)]
                if isinstance(weight_static, tf.Tensor):
                    weight_static = weight_static.numpy()
                res['feature_weight'] = dict(zip(self.evaluator.clause_features, weight_static))
            return res

        df = pd.json_normalize((empirical_result_to_record(d) for d in empirical_results), sep='_')
        df.set_index('problem', inplace=True)
        df = astype(df, {
            r'\w+_szs_status': pd.CategoricalDtype(szs.short_to_long),
            **{fr'\w+_{k}': pd.UInt32Dtype() for k in
               ['megainstructions', 'activations', 'memory', 'unique_feature_vectors', 'clauses_proof',
                'clauses_nonproof']}
        })
        log.debug('Empirical evaluation done.\n%s' % yaml.dump({
            'total': len(problems),
            'feature_weight_predicted': df.feature_weight_predicted.sum(),
            'probe': {
                **{col: df[f'probe_{col}'].sum() for col in ['solved', 'unsat']},
                'szs_status': {k: v for k, v in df.probe_szs_status.value_counts().items() if v > 0}
            }
        }))
        return df

    def evaluate_proxy(self, problems):
        batches = self.data.generate_batches(problems=problems, nonempty=True, exhaustive=True,
                                             join_searches=self.join_searches, **self.limits.predict)
        batch_stats = defaultdict(list)
        with tqdm(batches, unit='batch', desc='Proxy evaluation', disable=not get_verbose()) as t:
            for batch in t:
                if len(batch) == 0:
                    warnings.warn('Empty batch generated.')
                stats = self.test_step(batch)
                for k, v in stats.items():
                    batch_stats[k].append(v)
        data = {}
        for k, v in batch_stats.items():
            if k == 'proof_search':
                continue
            values = tf.concat(v, 0)
            if k == 'problem':
                data[k] = list(map(py_str, values))
                continue
            data[k] = values
        df = pd.DataFrame(data=data)
        if len(df) > 0:
            df.set_index('problem', inplace=True)
        records = []
        for pp, ps_stats in zip(batch_stats['problem'], batch_stats['proof_search']):
            for p, ps in zip(pp, ps_stats):
                for k, v in ps.items():
                    record = {
                        'problem': py_str(p),
                        'proof_search': k,
                        **{kk: v[kk].numpy() for kk in ['loss', 'accuracy']},
                    }
                    with suppress(KeyError):
                        record.update({'clauses_nonproof': v['clauses'][False], 'clauses_proof': v['clauses'][True]})
                    records.append(record)
        df_ps = pd.DataFrame.from_records(records, index=['problem', 'proof_search'])
        df_ps.sort_index(inplace=True)
        df_ps = astype(df_ps, {'clauses_nonproof': pd.UInt32Dtype(), 'clauses_proof': pd.UInt32Dtype()})
        return df, df_ps

    def train_step(self, batch):
        with tf.GradientTape() as tape:
            stats = self.test_step(batch, training=True)
            if stats is None:
                return None
            total_loss = tf.experimental.numpy.nansum(stats['loss'])
        self.minimize(total_loss, tape)
        self.update_stats({'train': {
            'loss': {
                'sum': total_loss,
                'mean': tf.experimental.numpy.nanmean(stats['loss'])
            },
            'accuracy': tf.experimental.numpy.nanmean(stats['accuracy'])
        }}, subset='train')

    def test_step(self, batch, training=False):
        if len(batch) == 0:
            return None
        dtypes = {'problem': tf.string, 'occurrence_count': tf.float32, 'nonproof': tf.bool,
                  'sample_weight': tf.float32}
        x = {
            'problem': to_tensor((sample['problem'] for sample in batch), dtype=dtypes.get('problem'), name='problem'),
            'occurrence_count': to_tensor((sample['clause_pairs']['X'] for sample in batch),
                                          dtype=dtypes.get('occurrence_count'),
                                          name='occurrence_count')
        }
        # We disable parallelization so that the graphification is not process-parallelized.
        # Such parallelization would be inefficient in case the graphs are retrieved from the disk cache.
        # We assume that they are almost always retrieved.
        with joblib.parallel_backend('threading', n_jobs=1):
            # We do not cache graphs because the problem set is sampled randomly.
            log.debug(f'Calling model on a batch. Number of problems: %u. Total size of the feature tensor: %u.\n%s' % (
                x['problem'].shape[0], tf.size(x['occurrence_count'], out_type=tf.uint32), pd.DataFrame.from_records(
                {
                    'problem': py_str(problem),
                    'clause_pairs': feature_matrix.shape[0],
                    'features': feature_matrix.row_lengths()[0].numpy(),
                    'size': tf.size(feature_matrix, out_type=tf.uint32).numpy()
                } for problem, feature_matrix in zip(x['problem'], x['occurrence_count']))))
            clause_pair_weights, problem_valid = self.model(x, training=training, expensive=False)

        weight_sum_tol = 1e-1

        def problem_sample_weight(clause_pairs):
            with suppress(KeyError):
                return aggregate_sample_weight(clause_pairs['sample_weight'])
            n = clause_pairs['X'].shape[0]
            if n > 0:
                value = 1 / n
            else:
                value = 0
            return np.full(n, value, dtype=dtypes['sample_weight'].as_numpy_dtype)

        sample_weight = to_tensor((problem_sample_weight(sample['clause_pairs']) for sample in batch),
                                  dtype=dtypes.get('sample_weight'), name='sample_weight')
        tf.debugging.assert_near(tf.ones(sample_weight.shape[0], dtype=sample_weight.dtype),
                                 tf.math.reduce_sum(sample_weight, axis=1),
                                 rtol=0, atol=weight_sum_tol)

        # Each row of `clause_pair_weights` is a difference of weights of a nonproof and a proof clause.
        # Nonproof clause weight should be large. Proof clause weight should b8e small.
        # "nonproof - proof" difference should be positive.
        with tf.name_scope('loss'):
            clause_pair_loss = -tf.math.log_sigmoid(clause_pair_weights, name='clause_pair')
            problem_loss = tf.reduce_sum(clause_pair_loss * tf.cast(sample_weight, clause_pair_loss.dtype), axis=1)
        clause_pair_hit = clause_pair_weights > 0
        problem_accuracy = tf.math.reduce_sum(tf.cast(clause_pair_hit, sample_weight.dtype) * sample_weight, axis=1,
                                              name='problem_accuracy')

        stats = {
            'problem': x['problem'],
            'loss': problem_loss,
            'accuracy': problem_accuracy,
            'valid': problem_valid
        }

        if not training:
            proof_search_stats = []
            for s, cpw, cpl in zip(batch, clause_pair_weights, clause_pair_loss):
                result = {}
                if 'sample_weight' in s['clause_pairs']:
                    sample_weight = s['clause_pairs']['sample_weight']
                    for k, col in sample_weight.items():
                        assert np.isclose(1, col.sum(), rtol=0, atol=weight_sum_tol)
                        result[k] = {
                            'loss': tf.math.reduce_sum(cpl * col, name='proof_search_loss'),
                            'accuracy': tf.math.reduce_sum(tf.cast(cpw > 0, dtypes['sample_weight']) * col)
                        }
                        if k in s['clause_pairs']['proof_search']:
                            result[k].update(s['clause_pairs']['proof_search'][k])
                else:
                    result['random_sample'] = {
                        'loss': tf.math.reduce_mean(cpl, name='proof_search_loss'),
                        'accuracy': tf.math.reduce_mean(tf.cast(cpw > 0, dtypes['sample_weight']))
                    }
                proof_search_stats.append(result)
            stats['proof_search'] = proof_search_stats

        return stats

    def minimize(self, loss, tape):
        assert ~tf.math.is_nan(loss)
        grads_and_vars = self.optimizer._compute_gradients(loss, var_list=self.model.trainable_weights, tape=tape)
        if all(grad is None for grad, var in grads_and_vars):
            warnings.warn('No gradient has been computed.')
        # Optimizing the model by a nan gradient invalidates the model.
        assert all(grad is None or tf.reduce_all(~tf.math.is_nan(grad)) for grad, var in grads_and_vars)
        return self.optimizer.apply_gradients((grad, var) for grad, var in grads_and_vars if grad is not None)

    @property
    def problems(self):
        return self.data.problems

    def stats(self, evaluation_stats):
        return {
            **self.data.stats,
            **{f'{k}_solved': v['problems']['solved'] / v['problems']['total'] for k, v in evaluation_stats.items()}
        }


class InitialDesign:
    def __init__(self, clausifier, clause_features, dist, random_state):
        self.clausifier = clausifier
        self.clause_features = clause_features
        self.dist = dist
        self.random_state = random_state

    def sample_weights(self, problems):
        return {problem: self.sample_weight(problem) for problem in problems}

    def sample_weight(self, problem):
        signature = self.clausifier.signature(problem)
        size = self.clause_features + len(signature)
        log.debug(f'{problem}: Sampling random weight vector of length {size}.')
        return self.dist.rvs(size=size, random_state=self.random_state)


class Dataset:
    def __init__(self, problems, ss, subsets=None):
        self.rng = np.random.default_rng(ss.spawn(1)[0])
        assert len(problems) == len(set(problems))
        self.problem_datasets = {problem: ProblemDataset(rng=np.random.default_rng(seed)) for problem, seed in
                                 zip(problems, ss.spawn(len(problems)))}
        self.subsets = subsets

    def __repr__(self):
        return str(self.stats)

    @property
    def stats(self):
        return {
            'problems': len(self.problems()),
            'problems_with_clauses': len(self.problems(nonempty=True)),
            'proof_searches': sum(len(p.proof_searches) for p in self.problem_datasets.values())
        }

    def update(self, results, **kwargs):
        for result in results:
            if 'verbose' in result:
                problem = result['problem']
                self.problem_datasets[problem].update(result['verbose']['clause_feature_vectors'], **kwargs)

    def generate_batches(self, problems=None, max_batch_samples=None, max_batch_size=None, exhaustive=False,
                         max_sample_size=None, **kwargs):
        if max_batch_samples is None and max_batch_size is None:
            # If the batch size is unlimited, it doesn't make sense to sample indefinitely.
            exhaustive = True
        if max_sample_size is None or (max_batch_size is not None and max_sample_size > max_batch_size):
            max_sample_size = max_batch_size
        samples = self.generate_samples(problems, exhaustive=exhaustive, max_size=max_sample_size, **kwargs)
        batch = []
        batch_size = 0
        for sample in samples:
            sample_size = np.prod(sample['clause_pairs']['X'].shape)
            if sample_size == 0:
                continue
            if (max_batch_samples is not None and len(batch) + 1 > max_batch_samples) or (
                    max_batch_size is not None and batch_size + sample_size > max_batch_size):
                yield batch
                batch = []
                batch_size = 0
            batch.append(sample)
            batch_size += sample_size
        if len(batch) > 0:
            yield batch

    def generate_samples(self, problems=None, exhaustive=False, max_size=None, join_searches=False, **kwargs):
        problems = self.problems(problems=problems, **kwargs)
        if not exhaustive:
            problems = self.sample_problems(problems)
        return (self.generate_sample(problem, max_size=max_size, join_searches=join_searches) for problem in problems)

    def sample_problems(self, problems):
        while True:
            yield self.rng.choice(problems)

    def generate_sample(self, problem, **kwargs):
        return {'problem': problem, 'clause_pairs': self.problem_datasets[problem].generate_batch(**kwargs)}

    def problems(self, problems=None, subset=None, nonempty=False):
        def is_relevant(problem):
            assert problem in self.problem_datasets
            if subset is not None and problem not in self.subsets[subset]:
                return False
            if nonempty and not self.problem_datasets[problem].has_clause_pairs:
                return False
            return True

        if problems is None:
            problems = self.problem_datasets
        return [str(problem) for problem in problems if is_relevant(problem)]

    def problem_stats(self, problems=None):
        if problems is None:
            problems = self.problems()
        records = [{'problem': problem, **self.problem_datasets[problem].record(),
                    'subset': {subset: problem in problems for subset, problems in self.subsets.items()}}
                   for problem in problems]
        df = pd.json_normalize(records, sep='_')
        df.set_index('problem', inplace=True)
        return df


class ProblemDataset:
    def __init__(self, rng=None):
        if rng is None:
            warnings.warn('Using default RNG.')
            rng = np.random.default_rng()
        self.rng = rng
        self.feature_vector_hash_to_index = {}
        self.feature_vectors = []
        self.proof_searches = {}
        self.proof_feature_vector_indices = set()

    def __len__(self):
        return len(self.feature_vectors)

    @property
    def has_clause_pairs(self):
        return any(
            any(v > 0 for v in proof_search[False].values()) and any(v > 0 for v in proof_search[True]) for proof_search
            in self.proof_searches.values())

    def __repr__(self):
        return str(self._asdict())

    def _asdict(self):
        return {
            'feature_vectors': len(self.feature_vectors),
            'proof_feature_vectors': len(self.proof_feature_vector_indices),
            'proof_seaches': len(self.proof_searches)
        }

    def record(self):
        return {
            'feature_vectors': len(self.feature_vectors),
            'proof_feature_vectors': len(self.proof_feature_vector_indices),
            'proof_searches': {
                'total': len(self.proof_searches),
                'with_proof_clauses': sum(len(ps[True]) > 0 for ps in self.proof_searches.values()),
                'with_nonproof_clauses': sum(len(ps[False]) > 0 for ps in self.proof_searches.values())
            }
        }

    def update(self, p, step):
        if len(p) == 0:
            raise ValueError('Empty proof search.')
        proof_search = {role_proof: Counter() for role_proof in [False, True]}
        for feature_vector_hash, v in p.items():
            assert all(vv >= 0 for vv in v['role_proof'].values())
            assert any(vv > 0 for vv in v['role_proof'].values())
            if feature_vector_hash not in self.feature_vector_hash_to_index:
                self.feature_vector_hash_to_index[feature_vector_hash] = len(self.feature_vector_hash_to_index)
                self.feature_vectors.append(v['feature_vector'])
            feature_vector_index = self.feature_vector_hash_to_index[feature_vector_hash]
            assert sparse_equal(self.feature_vectors[feature_vector_index], v['feature_vector'])
            for role_proof in [False, True]:
                if v['role_proof'][role_proof] == 0:
                    continue
                proof_search[role_proof][feature_vector_index] += v['role_proof'][role_proof]
        assert len(self.feature_vector_hash_to_index) == len(self.feature_vectors)
        assert step not in self.proof_searches
        self.proof_searches[step] = proof_search
        self.proof_feature_vector_indices.update(proof_search[True])

    def generate_batch(self, max_pairs=None, max_size=None, **kwargs):
        ds = dataset.proofs_to_samples(self.feature_vectors, self.proof_searches, **kwargs)
        n_pairs, n_features = ds['X'].shape
        if max_size is not None:
            max_pairs_from_size = max_size // n_features
            if max_pairs is None:
                max_pairs = max_pairs_from_size
            else:
                max_pairs = min(max_pairs, max_pairs_from_size)
        if max_pairs is not None and n_pairs > max_pairs:
            log.debug(f'Subsampling clause pairs. Before: {n_pairs}. After: {max_pairs}.')
            p = aggregate_sample_weight(ds['sample_weight'])
            chosen_indices = self.rng.choice(n_pairs, size=max_pairs, p=p)
            ds['X'] = ds['X'][chosen_indices]
            del ds['sample_weight']
        return ds


def aggregate_sample_weight(sample_weight):
    assert np.allclose(1, sample_weight.sum(), rtol=0)
    if isinstance(sample_weight, pd.DataFrame):
        sample_weight = sample_weight.mean(axis=1)
    assert np.isclose(1, sample_weight.sum(), rtol=0)
    return sample_weight
