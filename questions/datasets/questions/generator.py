import collections
import functools
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
        return self.df.loc[:, 'attempts']

    @property
    def problem_hits(self):
        return self.df.loc[:, 'hits']

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

    def load_questions(self, dir, num_questions_per_problem=None, simple=True):
        results = collections.defaultdict(list)
        for step in tqdm(range(self.step), desc='Loading question batches', unit='batch'):
            filename = os.path.join(dir, f'{step}.joblib')
            for problem_i, question in joblib.load(filename):
                problem_name = self.problems[problem_i]
                if num_questions_per_problem is None or len(results[problem_name]) < num_questions_per_problem:
                    if simple:
                        assert len(self.randomize) == 1
                        symbol_type = self.randomize[0]
                        precedences = (question['precedences'][i][symbol_type] for i in range(2))
                        precedences_inverted = tuple(
                            map(functools.partial(utils.invert_permutation, dtype=np.int32), precedences))
                        res = precedences_inverted[1] - precedences_inverted[0]
                        results[problem_name].append(res)
                    else:
                        results[problem_name].append(question)
        if simple:
            results = {k: np.asarray(v) for k, v in results.items()}
        return results

    def generate(self, solver, num_questions_per_batch=1000, num_questions_per_problem=None, num_questions=None,
                 dir=None, scatter_period=1000):
        questions_dir = os.path.join(dir, 'questions')
        os.makedirs(questions_dir, exist_ok=True)
        while num_questions is None or self.num_hits < num_questions:
            tf.summary.experimental.set_step(self.step)
            if num_questions_per_problem is not None and np.all(self.problem_hits >= num_questions_per_problem):
                logging.info('All problems have been saturated.')
                break
            batch = []
            bootstrap_batch = self.num_attempts == 0
            if bootstrap_batch:
                cur_batch_size = self.num_problems
            else:
                cur_batch_size = num_questions_per_batch
            for _ in range(cur_batch_size):
                if bootstrap_batch:
                    best = np.argmin(self.problem_attempts)
                else:
                    problem_ucbs = self.problem_ucbs()
                    if num_questions_per_problem is not None:
                        problem_ucbs[self.problem_hits.to_numpy() >= num_questions_per_problem] = np.NINF
                    best = np.argmax(problem_ucbs)
                # We specify the case number uniquely across problems.
                # If we maintained case id for each problem independently,
                batch.append((best, self.problem_attempts[best]))
                self.problem_attempts[best] += 1
            logging.info(f'Generating {len(batch)} questions...')
            questions = Parallel(verbose=1)(
                delayed(self.generate_one)(problem_i, case, solver) for problem_i, case in batch)
            result = [(problem_i, question) for (problem_i, case), question in zip(batch, questions) if
                      question is not None]
            for problem_i, question in result:
                self.problem_hits[problem_i] += 1
            if dir is not None:
                joblib.dump(result, os.path.join(questions_dir, f'{self.step}.joblib'))
                self.save(dir)
            logging.info(
                f'Step {self.step}: Problems with at least one question: {np.sum(self.problem_hits >= 1)}/{self.num_problems}. Total questions: {self.num_hits}/{self.num_attempts}/{num_questions}.')
            tf.summary.scalar('num_problems_with_questions', np.sum(self.problem_hits >= 1))
            tf.summary.scalar('num_questions', self.num_hits)
            tf.summary.scalar('num_attempts', self.num_attempts)
            tf.summary.scalar('batch_questions', sum(question is not None for question in questions))
            tf.summary.histogram('ucbs', self.problem_ucbs())
            tf.summary.histogram('attempts', self.problem_attempts.astype(np.uint32))
            tf.summary.histogram('hits', self.problem_hits.astype(np.uint32))
            tf.summary.histogram('hit_rates', self.problem_mean_rewards)
            tf.summary.histogram('confidence_margins', self.problem_ucbs() - self.problem_mean_rewards)
            if self.step % scatter_period == 0:
                plot.scatter(self.df[f'predicates'], self.df[f'functions'], name=f'problems/predicates_functions',
                             xlabel='predicates', ylabel='functions', xscale='log', yscale='log')
                plot.scatter(self.problem_attempts, self.problem_hits, name=f'problems/attempts_hits',
                             xlabel='Attempts', ylabel='Hits', xscale='log', yscale='log')
                for symbol_type in symbol_types:
                    x_col = f'{symbol_type}s'
                    x = self.df[x_col]
                    plot.scatter(x, self.problem_attempts, name=f'problems_{x_col}/attempts',
                                 xlabel=x_col, ylabel='Attempts', xscale='log', yscale='log')
                    plot.scatter(x, self.problem_hits, name=f'problems_{x_col}/hits',
                                 xlabel=x_col, ylabel='Hits', xscale='log', yscale='log')
                    plot.scatter(x, self.problem_mean_rewards, name=f'problems_{x_col}/hit_rates',
                                 xlabel=x_col, ylabel='Hit rate', xscale='log')
                    plot.scatter(x, self.problem_ucbs(), name=f'problems_{x_col}/ucbs',
                                 xlabel=x_col, ylabel='UCB', xscale='log')
            self.step += 1
        return self.load_questions(questions_dir)

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
    # Sometimes a failed attempt (`returncode==1`) does not output saturation iteration count.
    assert r0.returncode != 0 or r0.saturation_iterations is not None
    assert r1.returncode != 0 or r1.saturation_iterations is not None
    if r0.saturation_iterations is not None and r1.saturation_iterations is not None:
        if r0.returncode == 0 and r1.returncode != 0 and r0.saturation_iterations <= r1.saturation_iterations:
            return True
        if r0.returncode == 0 and r1.returncode == 0 and r0.saturation_iterations < r1.saturation_iterations:
            return True
    return False
