import collections
import functools
import itertools
import logging
import os
import re
import warnings

import joblib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from proving import utils
from proving.memory import memory


def load(file, questions_dir, max_questions_per_problem=None):
    if file is not None:
        logging.info(f'Loading questions from file {file}...')
        try:
            questions_all = joblib.load(file)
            logging.info(f'Questions loaded from file {file}.')
        except FileNotFoundError:
            logging.info(f'Failed to load questions from file {file}.')
            questions_all = get_problem_questions(questions_dir, max_questions_per_problem)
            logging.info(f'Saving questions to file {file}...')
            os.makedirs(os.path.dirname(file), exist_ok=True)
            joblib.dump(questions_all, file)
            logging.info(f'Questions saved to file {file}.')
    else:
        questions_all = get_problem_questions(questions_dir, max_questions_per_problem)
    logging.info(f'Number of problems with questions: {len(questions_all)}')
    return questions_all


def get_problem_questions(question_dir, max_questions_per_problem):
    # Collect question paths
    question_entry_list = get_question_paths(question_dir)
    logging.info(f'Total number of questions: {len(question_entry_list)}')

    # Limit the number of questions per problem
    question_path_lists = collections.defaultdict(list)
    n_questions_selected = 0
    for problem_name, question_path in tqdm(question_entry_list, unit='question', desc='Selecting questions to load'):
        if problem_name is not None and (max_questions_per_problem is None or len(
                question_path_lists[problem_name]) < max_questions_per_problem):
            question_path_lists[problem_name].append(question_path)
            n_questions_selected += 1
    logging.info(f'Number of questions selected: {n_questions_selected}')

    # Load questions
    queries = itertools.chain.from_iterable(question_path_lists.values())
    logging.info(f'Loading {n_questions_selected} questions...')
    question_list = Parallel(verbose=1)(delayed(load_question)(question_path) for question_path in queries)
    logging.info(f'Questions loaded. Number of questions loaded: {len(question_list)}')

    # Collect questions into a dictionary
    question_problem_names = itertools.chain.from_iterable(
        itertools.repeat(problem_name, len(vv)) for problem_name, vv in question_path_lists.items())
    question_lists = collections.defaultdict(list)
    for problem_name, question in tqdm(zip(question_problem_names, question_list), unit='question',
                                       desc='Sorting questions by problem', total=len(question_list)):
        assert problem_name is not None
        question_lists[problem_name].append(question)

    # Convert per-problem questions into an array
    question_arrays = {k: np.asarray(v) for k, v in tqdm(question_lists.items(), unit='problem',
                                                         desc='Converting questions to arrays')}

    return question_arrays


@memory.cache(verbose=2)
def get_question_paths(question_dir):
    def parse_question_dir_entry(dir_entry):
        m = re.search(r'^(?P<problem_name>[A-Z]{3}[0-9]{3}[-+^=_][1-9][0-9]*(\.[0-9]{3})*)_\d+\.q$', dir_entry.name,
                      re.MULTILINE)
        if m is None:
            warnings.warn(f'Failed to extract problem name from question file name: {dir_entry.name}')
            return None, None
        problem_name = m['problem_name']
        return problem_name, dir_entry.path

    logging.info(f'Parsing question paths in directory {question_dir}...')
    question_entry_list = Parallel(verbose=1)(
        delayed(parse_question_dir_entry)(dir_entry) for dir_entry in os.scandir(question_dir))
    logging.info(f'Question paths parsed. Number of questions: {len(question_entry_list)}')
    return question_entry_list


def load_question(question_path, normalize=False, dtype=None):
    if dtype is None:
        if normalize:
            dtype = np.float32
        else:
            dtype = np.int32
    content = open(question_path).read()
    m = re.search(r'^(?P<precedence_0>[0-9,]+)\n(?P<precedence_1>[0-9,]+)\n(?P<polarity>[<>])$', content, re.MULTILINE)
    precedence_strings = (m['precedence_0'], m['precedence_1'])
    # Precedence must be loaded as integer array so that it can be inverted.
    precedences = map(precedence_from_string, precedence_strings)
    # We cast the data to the desired dtype while inverting.
    precedences_inverted = tuple(map(functools.partial(utils.invert_permutation, dtype=dtype), precedences))
    # We temporarily assume that precedence 0 is better than precedence 1.
    # Then precedence pair cost uses the term `precedences_inverted[1] - precedences_inverted[0]`.
    res = precedences_inverted[1] - precedences_inverted[0]
    assert m['polarity'] in {'<', '>'}
    if m['polarity'] == '>':
        # If precedence 1 is actually better than precedence 0, we invert the term.
        res *= -1
    # Now we have the term precedence_inverted_worse - precedence_inverted_better.
    if normalize:
        n = len(res)
        res = res * dtype(2 / (n * (n + 1)))
    assert len(res.shape) == 1
    assert res.dtype == dtype
    assert np.isclose(0, res.sum(), atol=1e-06)
    # logging.debug(f'n={n}, abs.sum={np.sum(np.abs(res))}, abs.std={np.std(np.abs(res))}, std={np.std(res)}')
    return res


def precedence_from_string(s, dtype=np.uint32):
    return np.fromstring(s, sep=',', dtype=dtype)
