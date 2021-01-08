import collections
import logging
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels.stats.proportion
import tensorflow as tf
from joblib import delayed, Parallel

from proving import vampire
from proving.memory import memory
from proving.utils import dataframe_from_records
from questions import plot
from vampire_ml.results import save_df

symbol_types = ('predicate', 'function')


class Generator:
    def __init__(self, df, randomize=None, problem_questions=None, ucb_method='hoeffding', hoeffding_exponent=4):
        # The higher the exponent, the more exploration. The value of 4 corresponds to UCB1.
        self.df = df
        if randomize is None:
            randomize = symbol_types
        self.randomize = randomize
        if problem_questions is None:
            problem_questions = collections.defaultdict(list)
        self.problem_questions = problem_questions
        self.ucb_method = ucb_method
        self.hoeffding_exponent = hoeffding_exponent

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
        joblib.dump(self.problem_questions, os.path.join(dir, 'questions.joblib'))

    @classmethod
    def load(cls, dir):
        return joblib.load(os.path.join(dir, 'generator.joblib'))

    @property
    def num_attempts(self):
        return int(np.sum(self.problem_attempts))

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

    def generate(self, solver, num_questions_per_batch=1000, num_questions_per_problem=1000, num_questions=None,
                 dir=None):
        step = 0
        while num_questions is None or self.num_attempts < num_questions:
            tf.summary.experimental.set_step(step)
            if np.all(self.problem_hits >= num_questions_per_problem):
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
                    problem_ucbs[self.problem_hits.to_numpy() >= num_questions_per_problem] = np.NINF
                    best = np.argmax(problem_ucbs)
                # We specify the case number uniquely across problems.
                # If we maintained case id for each problem independently,
                batch.append((best, self.problem_attempts[best]))
                self.problem_attempts[best] += 1
            logging.info(f'Generating {len(batch)} questions...')
            questions = Parallel(verbose=10)(
                delayed(self.generate_one)(problem_i, case, solver) for problem_i, case in batch)
            for (problem_i, case), question in zip(batch, questions):
                if question is not None:
                    self.problem_hits[problem_i] += 1
                    problem_name = self.problems[problem_i]
                    self.problem_questions[problem_name].append(question)
            logging.info(
                f'Problems with at least one question: {len(self.problem_questions)}. Total questions: {np.sum(self.problem_hits)}/{self.num_attempts}/{num_questions}.')
            tf.summary.scalar('num_problems_with_questions', len(self.problem_questions))
            tf.summary.scalar('num_questions', np.sum(self.problem_hits))
            tf.summary.scalar('num_attempts', self.num_attempts)
            tf.summary.scalar('batch_questions', sum(question is not None for question in questions))
            tf.summary.histogram('ucbs', self.problem_ucbs())
            tf.summary.histogram('attempts', self.problem_attempts.astype(np.uint32))
            tf.summary.histogram('hits', self.problem_hits.astype(np.uint32))
            tf.summary.histogram('hit_rates', self.problem_mean_rewards)
            tf.summary.histogram('confidence_margins', self.problem_ucbs() - self.problem_mean_rewards)
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
            if dir is not None:
                self.save(dir)
            step += 1
        return self.problem_questions

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
