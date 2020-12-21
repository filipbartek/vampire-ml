import collections
import logging
import warnings

import joblib
import numpy as np
import tensorflow as tf
from joblib import delayed

from proving import vampire
from proving.memory import memory

symbol_types = ('predicate', 'function')


def generate(clausifier, solver, parallel, problems, num_questions=None, num_questions_per_batch=32, output=None):
    signature_sizes = get_signature_sizes(problems, clausifier, parallel)

    def generate_one_question(problem_i, case):
        signature_size = signature_sizes[problem_i]
        if signature_size is None:
            return None
        # TODO: Pivot.
        problem_name = problems[problem_i]
        precedences = [
            {symbol_type: vampire.random_precedence(symbol_type=symbol_type, length=signature_size[symbol_type],
                                                    seed=(case, i)) for symbol_type
             in symbol_types} for i in (0, 1)]
        results = [solver.solve(problem_name, precedences[i]) for i in (0, 1)]
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

    num_problems = len(problems)
    problem_attempts = np.zeros(num_problems, dtype=np.uint32)
    problem_hits = np.zeros(num_problems, dtype=np.uint32)
    problem_ucbs = np.full(num_problems, np.inf, dtype=np.float)
    num_attempts = 0
    problem_questions = collections.defaultdict(list)
    try:
        while num_questions is None or num_attempts < num_questions:
            batch = []
            cur_batch_size = num_questions_per_batch
            if num_questions is not None:
                cur_batch_size = min(cur_batch_size, num_questions - num_attempts)
            for case in range(num_attempts, num_attempts + cur_batch_size):
                best = np.argmax(problem_ucbs)
                # We specify the case number uniquely across problems.
                # If we maintained case id for each problem independently,
                batch.append((best, case))
                problem_attempts[best] += 1
                num_attempts += 1
                # https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-upper-confidence-bound-algorithm-4b84be516047
                problem_ucbs = problem_hits / problem_attempts + np.sqrt(np.log(num_attempts) / problem_attempts)
                problem_ucbs = np.nan_to_num(problem_ucbs, nan=np.inf)
            logging.info(f'Generating {len(batch)} questions...')
            questions = parallel(delayed(generate_one_question)(problem_i, case) for problem_i, case in batch)
            for (problem_i, case), question in zip(batch, questions):
                if question is not None:
                    problem_hits[problem_i] += 1
                    problem_name = problems[problem_i]
                    problem_questions[problem_name].append(question)
            logging.info(f'Problems with a question: {len(problem_questions)}. Total questions: {np.sum(problem_hits)}/{num_attempts}/{num_questions}.')
            problem_ucbs = problem_hits / problem_attempts + np.sqrt(2 * np.log(num_attempts) / problem_attempts)
            problem_ucbs = np.nan_to_num(problem_ucbs, nan=np.inf)
            tf.summary.scalar('problems_with_questions', len(problem_questions))
            tf.summary.scalar('questions', np.sum(problem_hits))
            tf.summary.scalar('attempts', num_attempts)
            tf.summary.scalar('batch_questions', sum(question is not None for question in questions))
            tf.summary.histogram('ucbs', problem_ucbs)
            tf.summary.histogram('attempts', problem_attempts)
            tf.summary.histogram('hits', problem_hits)
    finally:
        if output is not None:
            joblib.dump(problem_questions, output)
    return problem_questions


@memory.cache(ignore=['parallel'])
def get_signature_sizes(problems, solver, parallel):
    def get_signature_size(problem_name):
        clausify_result = solver.clausify(problem_name, get_symbols=True, get_clauses=False)
        try:
            return {symbol_type: len(clausify_result.symbols_of_type(symbol_type)) for symbol_type in symbol_types}
        except AttributeError:
            warnings.warn(f'Failed to get signature of problem {problem_name}: {clausify_result}')
            return None

    logging.info(f'Collecting signature sizes of {len(problems)} problems...')
    return parallel(delayed(get_signature_size)(problem_name) for problem_name in problems)


def is_better(r0, r1):
    if r0.returncode == 0 and r1.returncode != 0 and r0.saturation_iterations <= r1.saturation_iterations:
        return True
    if r0.returncode == 0 and r1.returncode == 0 and r0.saturation_iterations < r1.saturation_iterations:
        return True
    return False
