import logging
import warnings
from collections import Counter
from collections import defaultdict
from contextlib import suppress
from itertools import islice

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from more_itertools import chunked
from tqdm import tqdm

from questions.utils import flatten_dict
from questions.utils import py_str
from utils import astype
from utils import get_verbose
from utils import range_count
from utils import sparse_equal
from utils import to_tensor
from vampire import szs
from weight import dataset

log = logging.getLogger(__name__)


class Training:
    def __init__(self, dataset, model, optimizer=None, initial_design=None, evaluator=None, writers=None, epochs=None,
                 steps_per_epoch=10, train_batch_size=4, predict_batch_size=32):
        self.data = dataset
        self.model = model
        self.optimizer = optimizer
        self.initial_design = initial_design
        self.evaluator = evaluator
        self.writers = writers
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size

    def run(self):
        tf.summary.experimental.set_step(-1)
        epochs = range_count(self.epochs)
        with tqdm(epochs, unit='epoch', desc='Training', disable=not get_verbose()) as t:
            evaluation_stats = self.evaluate(initial=True)
            t.set_postfix(evaluation_stats)
            for step, epoch in enumerate(t):
                tf.summary.experimental.set_step(step)
                self.train_epoch(epoch)
                evaluation_stats = self.evaluate()
                t.set_postfix(evaluation_stats)

    def train_epoch(self, epoch):
        batches = islice(self.data.generate_batches(subset='train', nonempty=True, size=self.train_batch_size),
                         self.steps_per_epoch)
        with tqdm(batches, unit='step', total=self.steps_per_epoch, desc=f'Epoch {epoch}',
                  disable=not get_verbose()) as t, self.writers['train'].as_default():
            for step, batch in enumerate(t):
                if len(batch) == 0:
                    warnings.warn('Empty batch generated for training.')
                stats = self.train_step(batch)
                if stats is not None:
                    stats_public = {k: tf.experimental.numpy.nanmean(stats[k]) for k in ['loss', 'accuracy']}
                    for k, v in stats_public.items():
                        tf.summary.scalar(f'batch/{k}', v, step=epoch * self.steps_per_epoch + step)
                    t.set_postfix({k: v.numpy() for k, v in stats_public.items()})

    def evaluate(self, problems=None, initial=False):
        if problems is None:
            problems = self.problems()
        dfs = [self.data.problem_stats(problems),
               self.evaluate_proxy(problems),
               self.evaluate_empirical(problems, initial=initial)]
        df = pd.concat(dfs, axis='columns', copy=False)
        for col in ['feature_weight_predicted', 'probe_solved', 'probe_unsat', 'verbose_solved', 'verbose_unsat']:
            if col in df:
                df[col].fillna(False, inplace=True)
        stats = {}
        for subset, problems in self.data.subsets.items():
            cur_df = df[df.index.isin(problems)]
            with self.writers[subset].as_default():
                problem_stats = {
                    'solved': cur_df.probe_solved.mean(),
                    'attempted': cur_df.feature_weight_predicted.mean(),
                    'with_proof_clauses': (cur_df.proof_feature_vectors > 0).mean(),
                    'with_clauses': (cur_df.feature_vectors > 0).mean(),
                }
                for k, v in problem_stats.items():
                    tf.summary.scalar(f'problems/{k}', v)
                record = {}
                if 'loss' in cur_df:
                    record['loss'] = cur_df.loss.mean()
                if 'accuracy' in cur_df:
                    record['accuracy'] = cur_df.accuracy.mean()
                for k, v in record.items():
                    tf.summary.scalar(k, v)
                record['solved'] = problem_stats['solved']
                stats[subset] = record
        return flatten_dict(stats, sep='_')

    def evaluate_empirical(self, problems=None, initial=False):
        if initial and self.initial_design is not None:
            problem_weights = self.initial_design.sample_weights(problems)
        else:
            print(f'Empirical evaluation: Predicting clause feature weights for {len(problems)} problems...')
            model_result = self.model.symbol_weight_model.predict(problems, batch_size=self.predict_batch_size)
            # Contains two entries: valid, costs
            problem_weights = {}
            for problem, cost, valid in zip(problems, model_result['costs'], model_result['valid']):
                if valid:
                    problem_weights[problem] = cost
                else:
                    problem_weights[problem] = None
        empirical_results = self.evaluator.evaluate(problem_weights)
        self.data.update(empirical_results)

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
                res['feature_weight'] = {k: v.numpy() for k, v in zip(self.evaluator.clause_features, weight)}
            return res

        df = pd.json_normalize((empirical_result_to_record(d) for d in empirical_results), sep='_')
        df.set_index('problem', inplace=True)
        df = astype(df, {
            r'\w+_szs_status': pd.CategoricalDtype(szs.short_to_long),
            **{fr'\w+_{k}': pd.UInt32Dtype() for k in
               ['megainstructions', 'activations', 'memory', 'unique_feature_vectors', 'clauses_proof',
                'clauses_nonproof']}
        })
        return df

    def evaluate_proxy(self, problems):
        batches = self.data.generate_batches(problems=problems, nonempty=True, size=self.predict_batch_size,
                                             exhaustive=True)
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
            values = tf.concat(v, 0)
            if k == 'problem':
                data[k] = list(map(py_str, values))
                continue
            data[k] = values
        df = pd.DataFrame(data=data)
        if len(df) > 0:
            df.set_index('problem', inplace=True)
        return df

    def train_step(self, batch):
        with tf.GradientTape() as tape:
            stats = self.test_step(batch, training=True)
            if stats is None:
                return None
            total_loss = tf.experimental.numpy.nansum(stats['loss'])
        self.minimize(total_loss, tape)
        return stats

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
            log.debug(f'Calling model on a batch. Number of problems: %u. Total size of the feature tensor: %u.' % (
                x['problem'].shape[0], tf.size(x['occurrence_count'], out_type=tf.uint32)))
            log.debug('Calling model on a batch:\n%s' % pd.DataFrame.from_records(
                {
                    'problem': py_str(problem),
                    'clause_pairs': feature_matrix.shape[0],
                    'features': feature_matrix.row_lengths()[0].numpy(),
                    'size': tf.size(feature_matrix, out_type=tf.uint32).numpy()
                } for problem, feature_matrix in zip(x['problem'], x['occurrence_count'])))
            clause_pair_weights, problem_valid = self.model(x, training=training, expensive=False)
        # Each row of `clause_pair_weights` is a difference of weights of a nonproof and a proof clause.
        # Nonproof clause weight should be large. Proof clause weight should be small.
        # "nonproof - proof" difference should be positive.
        with tf.name_scope('loss'):
            assert all(
                'sample_weight' not in s['clause_pairs'] or np.isclose(1, s['clause_pairs']['sample_weight'].sum()) for
                s in batch)

            def problem_sample_weight(clause_pairs):
                with suppress(KeyError):
                    return clause_pairs['sample_weight']
                n = clause_pairs['X'].shape[0]
                if n > 0:
                    return tf.fill((n,), 1 / n, name='sample_weight_uniform')
                return tf.zeros((n,), name='sample_weight_empty')

            sample_weight = to_tensor((problem_sample_weight(sample['clause_pairs']) for sample in batch),
                                      dtype=dtypes.get('sample_weight'), name='sample_weight')
            tf.debugging.assert_near(tf.ones(sample_weight.shape[0]), tf.math.reduce_sum(sample_weight, axis=1),
                                     atol=0.5, rtol=0)
            clause_pair_loss = -tf.math.log_sigmoid(clause_pair_weights, name='clause_pair')
            problem_loss = tf.reduce_sum(clause_pair_loss * sample_weight, axis=1)
        clause_pair_hit = clause_pair_weights > 0
        problem_accuracy = tf.math.reduce_sum(tf.cast(clause_pair_hit, sample_weight.dtype) * sample_weight, axis=1,
                                              name='problem_accuracy')
        return {
            'problem': x['problem'],
            'loss': problem_loss,
            'accuracy': problem_accuracy
        }

    def minimize(self, loss, tape):
        grads_and_vars = self.optimizer._compute_gradients(loss, var_list=self.model.trainable_weights, tape=tape)
        if all(grad is None for grad, var in grads_and_vars):
            warnings.warn('No gradient has been computed.')
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
    def __init__(self, clausifier, samples_per_problem, clause_features, dist, random_state):
        self.clausifier = clausifier
        self.samples_per_problem = samples_per_problem
        self.clause_features = clause_features
        self.dist = dist
        self.random_state = random_state

    def sample_weights(self, problems):
        return {problem: self.sample_weight(problem) for problem in problems}

    def sample_weight(self, problem):
        signature = self.clausifier.signature(problem)
        size = (self.samples_per_problem, self.clause_features + len(signature))
        log.debug(f'{problem}: Sampling {size[0]} random weight vectors of length {size[1]}.')
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

    def update(self, results):
        for result in results:
            if 'verbose' in result:
                problem = result['problem']
                self.problem_datasets[problem].update(result['verbose']['clause_feature_vectors'])

    def generate_batches(self, problems=None, size=None, **kwargs):
        if size is None:
            while True:
                yield list(self.generate_samples(problems, **kwargs, exhaustive=True))
        else:
            for batch in chunked(self.generate_samples(problems, **kwargs), size):
                yield batch

    def generate_samples(self, problems=None, exhaustive=False, **kwargs):
        problems = self.problems(problems=problems, **kwargs)
        if not exhaustive:
            problems = self.sample_problems(problems)
        return (self.generate_sample(problem) for problem in problems)

    def sample_problems(self, problems):
        while True:
            yield self.rng.choice(problems)

    def generate_sample(self, problem):
        return {'problem': problem, 'clause_pairs': self.problem_datasets[problem].generate_batch()}

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
        self.proof_searches = []
        self.proof_feature_vector_indices = set()

    def __len__(self):
        return len(self.feature_vectors)

    @property
    def has_clause_pairs(self):
        return any(
            any(v > 0 for v in proof_search[False].values()) and any(v > 0 for v in proof_search[True]) for proof_search
            in self.proof_searches)

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
                'with_proof_clauses': sum(len(ps[True]) > 0 for ps in self.proof_searches),
                'with_nonproof_clauses': sum(len(ps[False]) > 0 for ps in self.proof_searches)
            }
        }

    def update(self, p):
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
        self.proof_searches.append(proof_search)
        self.proof_feature_vector_indices.update(proof_search[True])

    def generate_batch(self, max_pairs=None, max_sample_size=None):
        # TODO: Fix negatives in failed proof searches.
        ds = dataset.proofs_to_samples(self.feature_vectors, self.proof_searches)
        n_samples, n_features = ds['X'].shape
        if max_sample_size is not None:
            new_max_pairs = max_sample_size // n_features
            if max_pairs is None:
                max_pairs = new_max_pairs
            else:
                max_pairs = min(max_pairs, new_max_pairs)
        if max_pairs is not None and n_samples > max_pairs:
            log.debug(f'Subsampling clause pairs. Before: {n_samples}. After: {max_pairs}.')
            chosen_indices = self.rng.choice(n_samples, size=max_pairs, p=ds['sample_weight'])
            ds['X'] = ds['X'][chosen_indices]
            del ds['sample_weight']
        return ds
