import collections
import functools
import itertools
import logging
import os
import pickle
import re
import warnings

import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from ordered_set import OrderedSet
from tqdm import tqdm

from proving import symbols
from proving import utils
from proving.memory import memory
from vampire_ml.results import save_df

dtype_tf_float = np.float32

Question = collections.namedtuple('Question', ('problem_name', 'ranking_difference'))
ProblemSample = collections.namedtuple('ProblemSample', ('problem_name', 'ranking_differences'))


def get_ds_problems(question_dir, cache_file=None):
    file_pattern = os.path.join(question_dir, '*.q')
    ds_filenames = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    ds_general_questions = ds_filenames.map(filename_to_general_question).cache(cache_file)
    problem_names, questions = zip(*tqdm(iter(ds_general_questions), total=len(ds_general_questions), desc='Loading questions', unit='question'))
    y, idx = tf.unique(problem_names)
    return tf.data.Dataset.from_generator(functools.partial(generate_problem_samples, y, idx, questions),
                                          (tf.string, tf.int32)), len(y)


@tf.function
def generate_problem_samples(y, idx, questions):
    for problem_i, problem_name in enumerate(y):
        problem_question_indices = tf.where(idx == problem_i)[:, 0]
        problem_questions = [questions[i] for i in problem_question_indices]
        yield ProblemSample(problem_name, tf.stack(problem_questions))


@tf.function
def filename_to_general_question(filename):
    return Question(filename_to_problem_name(filename), filename_to_question(filename))


@tf.function
def filename_to_problem_name(filename):
    return tf.strings.regex_replace(filename,
                                    r'^.*\b(?P<problem_name>[A-Z]{3}[0-9]{3}[-+^=_][1-9][0-9]*(?:\.[0-9]{3})*)_\d+\.q$',
                                    r'\1', False)


def filename_to_question(filename):
    it = iter(tf.data.TextLineDataset(filename))
    precedences = tuple(map(line_to_precedence, itertools.islice(it, 2)))
    precedence_diff = precedences[1] - precedences[0]
    polarity = next(it)
    res = tf.cond(polarity == '>', lambda: precedence_diff * -1, lambda: precedence_diff)
    return res


def line_to_precedence(x):
    return tf.strings.to_number(tf.strings.split(x, sep=','), out_type=tf.int32)


@memory.cache(ignore=['cache_file', 'output_dir'], verbose=2)
def get_problems(question_dir, signature_dir, graphifier, max_problems, max_questions_per_problem, rng, cache_file, output_dir):
    if cache_file is not None:
        try:
            logging.info(f'Loading problems from {cache_file}...')
            problems = pickle.load(open(cache_file, mode='rb'))
            logging.info(f'Problems loaded from {cache_file}.')
            return problems
        except FileNotFoundError:
            pass
    questions = get_problem_questions(question_dir, rng, max_problems=max_problems,
                                      max_questions_per_problem=max_questions_per_problem)
    problem_names = list(questions.keys())
    signatures = get_problem_signatures(signature_dir, problem_names)
    graphs_records = graphifier.problems_to_graphs(problem_names)
    graphs, records = zip(*graphs_records)
    if output_dir is not None:
        save_df(utils.dataframe_from_records(records, index_keys='problem'), 'graphs', output_dir)
    problems = {problem_name: {'graph': graphs[i], 'questions': questions[problem_name], 'signatures': signatures[problem_name]} for
                i, problem_name in enumerate(problem_names) if graphs[i] is not None}
    if cache_file is not None:
        logging.info(f'Saving problems into {cache_file}...')
        pickle.dump(problems, open(cache_file, mode='wb'))
        logging.info(f'Problems saved into {cache_file}.')
    return problems


@memory.cache(verbose=2)
def get_problem_signatures(symbols_dir_path, problems=None):
    if problems is not None:
        iterable = ((problem_name, os.path.join(symbols_dir_path, f'{problem_name}.sig')) for problem_name in problems)
        total = len(problems)
    else:
        iterable = ((os.path.splitext(dir_entry.name)[0], dir_entry.path) for dir_entry in os.scandir(symbols_dir_path))
        total = None
    signatures = collections.defaultdict(dict)
    with tqdm(iterable, unit='problem', desc='Loading signatures', total=total) as t:
        for problem_name, signature_path in t:
            t.set_postfix_str(signature_path)
            try:
                sym_all = symbols.load(signature_path)
                for symbol_type in ('predicate', 'function'):
                    sym_selected = symbols.symbols_of_type(sym_all, symbol_type)
                    signatures[problem_name][symbol_type] = sym_selected.drop('name', axis='columns').astype(dtype_tf_float).values
            except ValueError:
                warnings.warn(f'Failed to load signature: {signature_path}')
    return signatures


@memory.cache(verbose=2)
def get_problem_questions(question_dir, rng, max_problems=None, max_questions_per_problem=None):
    # Parse paths
    question_entry_list = get_question_paths(question_dir)

    # Ensure we have at most max_problems problems
    problem_names = OrderedSet(tuple(zip(*question_entry_list))[0])
    logging.info(f'Total number of problems: {len(problem_names)}')
    if max_problems is not None and len(problem_names) > max_problems:
        problem_names = rng.choice(problem_names, size=max_problems, replace=False)

    # Filter problems with too many questions
    question_paths = {problem_name: [] for problem_name in problem_names}
    for problem_name, question_path in question_entry_list:
        if problem_name not in question_paths:
            continue
        if max_questions_per_problem is None or len(question_paths[problem_name]) < max_questions_per_problem:
            question_paths[problem_name].append(question_path)

    # Load questions
    queries = itertools.chain.from_iterable(question_paths.values())
    logging.info('Loading questions...')
    question_list = Parallel(verbose=1)(delayed(load_question)(question_path) for question_path in queries)
    logging.info(f'Questions loaded. Number of questions: {len(question_list)}')

    # Collect questions into a dictionary
    question_problem_names = itertools.chain.from_iterable(
        itertools.repeat(problem_name, len(vv)) for problem_name, vv in question_paths.items())
    questions = {problem_name: [] for problem_name in question_paths.keys()}
    for problem_name, question in zip(question_problem_names, question_list):
        questions[problem_name].append(question)

    # Convert per-problem questions into an array
    for problem_name in questions:
        questions[problem_name] = np.asarray(questions[problem_name])

    return questions


@memory.cache(verbose=2)
def get_question_paths(question_dir):
    def parse_question_dir_entry(dir_entry):
        m = re.search(
            r'^(?P<problem_name>(?P<problem_domain>[A-Z]{3})(?P<problem_number>[0-9]{3})(?P<problem_form>[-+^=_])(?P<problem_version>[1-9])(?P<problem_size_parameters>[0-9]*(\.[0-9]{3})*))_(?P<question_number>\d+)\.q$',
            dir_entry.name, re.MULTILINE)
        problem_name = m['problem_name']
        return problem_name, dir_entry.path

    logging.info(f'Parsing question paths in directory {question_dir}...')
    question_entry_list = Parallel(verbose=1)(
        delayed(parse_question_dir_entry)(dir_entry) for dir_entry in os.scandir(question_dir))
    logging.info(f'Question paths parsed. Number of questions: {len(question_entry_list)}')
    return question_entry_list


def load_question(question_path, normalize=True, dtype=dtype_tf_float):
    content = open(question_path).read()
    m = re.search(r'^(?P<precedence_0>[0-9,]+)\n(?P<precedence_1>[0-9,]+)\n(?P<polarity>[<>])$', content, re.MULTILINE)
    precedence_strings = (m['precedence_0'], m['precedence_1'])
    precedences = map(precedence_from_string, precedence_strings)
    precedences_inverted = tuple(map(functools.partial(utils.invert_permutation, dtype=dtype), precedences))
    res = precedences_inverted[1] - precedences_inverted[0]
    assert m['polarity'] in {'<', '>'}
    if m['polarity'] == '>':
        res *= -1
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
