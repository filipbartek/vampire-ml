import collections
import functools
import itertools
import logging
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels.stats.proportion
import tensorflow as tf
from joblib import delayed, Parallel
from tqdm import tqdm

from proving import utils
from proving import vampire
from proving.memory import memory
from proving.utils import dataframe_from_records
from questions import plot
from vampire_ml.results import save_df

symbol_types = ('predicate', 'function')


class Generator:
    def __init__(self, df, randomize=None, ucb_method='hoeffding', hoeffding_exponent=4, step=0):
        # The higher the exponent, the more exploration. The value of 4 corresponds to UCB1.
        self.df = df
        if randomize is None:
            randomize = symbol_types
        self.randomize = randomize
        self.ucb_method = ucb_method
        self.hoeffding_exponent = hoeffding_exponent
        self.step = step

    name = 'generator'

    @classmethod
    def fresh(cls, problems, clausifier, randomize=None, ucb_method='hoeffding', hoeffding_exponent=4):
        signature_sizes = get_signature_sizes(problems, clausifier)
        assert len(signature_sizes) == len(problems)
        records = [{
            'problem': problems[i],
            'predicates': signature_sizes[i]['predicate'],
            'functions': signature_sizes[i]['function'],
            'attempts': 0,
            'hits': 0
        } for i in range(len(problems))]
        dtypes = {
            'problem': 'object',
            'predicates': pd.UInt32Dtype(),
            'functions': pd.UInt32Dtype(),
            'attempts': pd.UInt32Dtype(),
            'hits': pd.UInt32Dtype()
        }
        df = dataframe_from_records(records, index_keys='problem', dtypes=dtypes)
        return cls(df, randomize, ucb_method=ucb_method, hoeffding_exponent=hoeffding_exponent)

    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        joblib.dump(self, os.path.join(dir, 'generator.joblib'))
        save_df(self.df, os.path.join(dir, 'problems'))

    @classmethod
    def load(cls, dir):
        generator = joblib.load(os.path.join(dir, 'generator.joblib'))
        # The step stored in the generator is the last completed step.
        generator.step += 1
        return generator

    @property
    def num_attempts(self):
        return int(np.sum(self.problem_attempts))

    @property
    def num_hits(self):
        return np.sum(self.problem_hits)

    @property
    def problem_attempts(self):
        return self.df['attempts']

    @problem_attempts.setter
    def problem_attempts(self, value):
        self.df['attempts'] = value

    @property
    def problem_hits(self):
        return self.df['hits']

    @property
    def problem_mean_rewards(self):
        return self.problem_hits / self.problem_attempts

    @property
    def problems(self):
        return self.df.index

    @property
    def num_problems(self):
        return len(self.df)

    def problem_ucbs(self):
        # https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-upper-confidence-bound-algorithm-4b84be516047
        # https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html
        if self.ucb_method == 'hoeffding':
            with np.errstate(all='raise'):
                res = self.problem_mean_rewards + np.sqrt(
                    self.hoeffding_exponent * np.log(self.num_attempts) / (2 * self.problem_attempts))
                assert not np.any(np.isnan(res))
        else:
            # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
            ci_low, res = statsmodels.stats.proportion.proportion_confint(self.problem_hits.astype(np.uint32),
                                                                          self.problem_attempts.astype(np.uint32),
                                                                          method=self.ucb_method)
        return res

    def load_questions(self, questions_dir, num_questions_per_problem=None, num_questions=None):
        cache_filename = os.path.join(questions_dir,
                                      f'per_problem_{num_questions_per_problem}',
                                      f'count_{num_questions}',
                                      'questions.joblib')
        try:
            results = joblib.load(cache_filename)
            logging.info(f'Questions loaded from a cache file: {cache_filename}')
        except FileNotFoundError:
            results = collections.defaultdict(list)
            num_loaded = 0
            for step in tqdm(range(self.step), desc='Loading question batches', unit='batch'):
                if num_questions is not None and num_loaded >= num_questions:
                    break
                filename = os.path.join(questions_dir, f'{step}.joblib')
                for problem_i, attempt in joblib.load(filename):
                    if num_questions is not None and num_loaded >= num_questions:
                        break
                    problem_name = self.problems[problem_i]
                    if num_questions_per_problem is not None and len(results[problem_name]) >= num_questions_per_problem:
                        continue
                    question = self.get_question(*attempt)
                    if question is None:
                        continue
                    assert len(self.randomize) == 1
                    symbol_type = self.randomize[0]
                    precedences = (question['precedences'][i][symbol_type] for i in range(2))
                    # We assume that precedences[0] is better than precedences[1].
                    precedences_inverted = tuple(
                        map(functools.partial(utils.invert_permutation, dtype=np.int32), precedences))
                    res = precedences_inverted[1] - precedences_inverted[0]
                    results[problem_name].append(res)
                    num_loaded += 1
            results = {k: np.asarray(v) for k, v in results.items()}
            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
            joblib.dump(results, cache_filename)
            logging.info(f'Questions saved to a cache file: {cache_filename}')
        return results

    @staticmethod
    def get_question(precedences, results):
        # Ensure that the output precedence 0 is better than the output precedence 1.
        if is_better(results[0], results[1]):
            return {
                'precedences': precedences,
                'results': results
            }
        elif is_better(results[1], results[0]):
            return {
                'precedences': [precedences[1], precedences[0]],
                'results': [results[1], results[0]]
            }
        return None

    def generate(self, solver, num_questions_per_batch=1000, num_questions_per_problem=None, num_questions=None,
                 dir=None, scatter_period=10):
        questions_dir = None
        if dir is not None:
            questions_dir = os.path.join(dir, 'attempts')
            os.makedirs(questions_dir, exist_ok=True)
        while num_questions is None or self.num_hits < num_questions:
            tf.summary.experimental.set_step(self.step)
            if num_questions_per_problem is not None and np.all(self.problem_hits >= num_questions_per_problem):
                logging.info('All problems have been saturated.')
                break
            bootstrap_batch = self.num_attempts == 0
            if bootstrap_batch:
                if num_questions is not None:
                    # Bootstrap with multiple trials per problem. Use a lower bound on the final number of runs per problem. This gives better initial estimates of rewards.
                    # https://www.wolframalpha.com/input/?i=plot+%284+log%28q%29%29+%2F+%282+sqrt%281+%2B+%284+log%28q%29+p%29+%2F+%282+q%29%29%29%2C+q%3D1..1000000%2C+p%3D1..20000
                    c = self.hoeffding_exponent * np.log(num_questions)
                    bootstrap_copies = int(
                        np.ceil(c / (2 * np.square(1 + np.sqrt(c * self.num_problems / (2 * num_questions))))))
                else:
                    bootstrap_copies = 1
                logging.info(f'Bootstrapping with {bootstrap_copies} attempts per problem.')
                batch = [(problem_i, attempt_i) for attempt_i, problem_i in
                         itertools.product(range(bootstrap_copies), range(self.num_problems))]
                self.problem_attempts += bootstrap_copies
            else:
                batch = []
                for _ in range(num_questions_per_batch):
                    problem_ucbs = self.problem_ucbs()
                    if num_questions_per_problem is not None:
                        problem_ucbs[self.problem_hits.to_numpy() >= num_questions_per_problem] = np.NINF
                    best = np.argmax(problem_ucbs)
                    # We specify the case number uniquely across problems.
                    # If we maintained case id for each problem independently,
                    batch.append((best, self.problem_attempts[best]))
                    self.problem_attempts[best] += 1
            logging.info(f'Generating {len(batch)} questions...')
            attempts = Parallel(verbose=1)(
                delayed(self.generate_one)(problem_i, case, solver) for problem_i, case in batch)
            attempts_with_indices = list(zip(tuple(zip(*batch))[0], attempts))
            batch_hits = 0
            for problem_i, attempt in attempts_with_indices:
                if self.get_question(*attempt) is not None:
                    self.problem_hits[problem_i] += 1
                    batch_hits += 1
            if questions_dir is not None:
                joblib.dump(attempts_with_indices, os.path.join(questions_dir, f'{self.step}.joblib'))
            if dir is not None:
                self.save(dir)
            logging.info(
                f'Step {self.step}: Total problems hit: {np.sum(self.problem_hits >= 1)}/{self.num_problems}. Total hits: {self.num_hits}/{self.num_attempts}/{num_questions}.')
            tf.summary.scalar(f'{self.name}/total_problems_hit', np.sum(self.problem_hits >= 1))
            tf.summary.scalar(f'{self.name}/total_hits', self.num_hits)
            tf.summary.scalar(f'{self.name}/attempts/sum', self.num_attempts)
            tf.summary.scalar(f'{self.name}/attempts/min', self.problem_attempts.min())
            tf.summary.scalar(f'{self.name}/attempts/max', self.problem_attempts.max())
            tf.summary.scalar(f'{self.name}/total_hit_rate', self.num_hits / self.num_attempts)
            tf.summary.scalar(f'{self.name}/batch_hits', batch_hits)
            tf.summary.scalar(f'{self.name}/batch_hit_rate', batch_hits / len(batch))
            tf.summary.histogram(f'{self.name}/ucbs', self.problem_ucbs().astype(np.float64))
            tf.summary.histogram(f'{self.name}/attempts', self.problem_attempts.astype(np.uint32))
            tf.summary.histogram(f'{self.name}/hits', self.problem_hits.astype(np.uint32))
            tf.summary.histogram(f'{self.name}/hit_rates', self.problem_mean_rewards.astype(np.float64))
            tf.summary.histogram(f'{self.name}/ucb_margins',
                                 (self.problem_ucbs() - self.problem_mean_rewards).astype(np.float64))
            if self.step % scatter_period == 0:
                plot.scatter(self.df['predicates'], self.df['functions'], name=f'{self.name}/predicates/functions',
                             xlabel='predicates', ylabel='functions', xscale='log', yscale='log')
                plot.scatter(self.problem_attempts, self.problem_hits, name=f'{self.name}/attempts/hits',
                             xlabel='Attempts', ylabel='Hits', xscale='log', yscale='log')
                for symbol_type in symbol_types:
                    x_col = f'{symbol_type}s'
                    x = self.df[x_col]
                    plot.scatter(x, self.problem_attempts, name=f'{self.name}/{x_col}/attempts',
                                 xlabel=x_col, ylabel='Attempts', xscale='log', yscale='log')
                    plot.scatter(x, self.problem_hits, name=f'{self.name}/{x_col}/hits',
                                 xlabel=x_col, ylabel='Hits', xscale='log', yscale='log')
                    plot.scatter(x, self.problem_mean_rewards, name=f'{self.name}/{x_col}/hit_rates',
                                 xlabel=x_col, ylabel='Hit rate', xscale='log')
                    plot.scatter(x, self.problem_ucbs(), name=f'{self.name}/{x_col}/ucbs',
                                 xlabel=x_col, ylabel='UCB', xscale='log')
            self.step += 1
        return self.load_questions(questions_dir, num_questions_per_problem=num_questions_per_problem,
                                   num_questions=num_questions)

    def generate_one(self, problem_i, case, solver):
        problem_name = self.problems[problem_i]
        try:
            precedences = [{}, {}]
            for symbol_type in symbol_types:
                for i in range(2):
                    if symbol_type in self.randomize:
                        seed = (problem_i, case, i)
                    else:
                        seed = (problem_i, case, 0)
                    precedences[i][symbol_type] = vampire.random_precedence(symbol_type=symbol_type,
                                                                            length=self.signature_size(problem_i,
                                                                                                       symbol_type),
                                                                            seed=seed)
            results = [solver.solve(problem_name, precedences[i]) for i in range(2)]
            return precedences, results
        except Exception as e:
            raise RuntimeError(f'Failed to generate question {case} for problem {problem_name}.') from e

    def signature_size(self, problem_i, symbol_type):
        return self.df[f'{symbol_type}s'][problem_i]


@memory.cache(verbose=1)
def get_signature_sizes(problems, clausifier):
    def get_signature_size(problem_name):
        clausify_result = clausifier.clausify(problem_name, get_symbols=True, get_clauses=False)
        try:
            return {symbol_type: len(clausify_result.symbols_of_type(symbol_type)) for symbol_type in symbol_types}
        except AttributeError:
            warnings.warn(f'Failed to get signature of problem {problem_name}: {clausify_result}')
            return None

    logging.info(f'Collecting signature sizes of {len(problems)} problems...')
    return Parallel(verbose=10)(delayed(get_signature_size)(problem_name) for problem_name in problems)


def is_better(r0, r1):
    # Returns true if r0 is a better result than t1.
    # Sometimes a failed attempt (`returncode==1`) does not output saturation iteration count.
    assert r0.returncode != 0 or r0.saturation_iterations is not None
    assert r1.returncode != 0 or r1.saturation_iterations is not None
    if r0.saturation_iterations is not None and r1.saturation_iterations is not None:
        if r0.returncode == 0 and r1.returncode != 0 and r0.saturation_iterations <= r1.saturation_iterations:
            return True
        if r0.returncode == 0 and r1.returncode == 0 and r0.saturation_iterations < r1.saturation_iterations:
            return True
    return False
