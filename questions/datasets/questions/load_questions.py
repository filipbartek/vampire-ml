import collections
import functools
import logging
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from joblib import Parallel, delayed

from proving import utils
from questions import plot


def load(file, questions_dir):
    if file is not None:
        logging.info(f'Loading questions from file {file}...')
        try:
            questions_all = joblib.load(file)
            logging.info(f'Questions loaded from file {file}.')
        except FileNotFoundError:
            logging.info(f'Failed to load questions from file {file}.')
            questions_all = get_problem_questions(questions_dir)
            logging.info(f'Saving questions to file {file}...')
            joblib.dump(questions_all, file)
            logging.info(f'Questions saved to file {file}.')
    else:
        questions_all = get_problem_questions(questions_dir)
    logging.info(f'Number of problems with questions: {len(questions_all)}')
    question_counts = [q.shape[0] for q in questions_all.values()]
    signature_lengths = [q.shape[1] for q in questions_all.values()]
    tf.summary.histogram('Question counts', question_counts)
    tf.summary.histogram('Signature lengths of problems with some questions', signature_lengths)
    tf.summary.histogram('Question array sizes', [q.size for q in questions_all.values()])
    figure = plt.figure(figsize=(8, 8))
    plt.title('Problems with questions')
    sns.scatterplot(signature_lengths, question_counts)
    plt.xlabel('Symbols')
    plt.ylabel('Questions')
    plt.xscale('log')
    plt.yscale('log')
    image = plot.plot_to_image(figure)
    tf.summary.image('Problems with questions', image)
    return questions_all


def get_problem_questions(question_dir):
    def load_one(dir_entry):
        m = re.search(r'^(?P<problem_name>[A-Z]{3}[0-9]{3}[-+^=_][1-9][0-9]*(\.[0-9]{3})*)_\d+\.q$', dir_entry.name,
                      re.MULTILINE)
        problem_name = m['problem_name']
        question = load_question(dir_entry.path)
        return problem_name, question

    logging.info(f'Loading questions from directory {question_dir}...')
    question_entry_list = Parallel(verbose=1)(delayed(load_one)(dir_entry) for dir_entry in os.scandir(question_dir))
    logging.info(f'Questions loaded. Number of questions: {len(question_entry_list)}')

    question_lists = collections.defaultdict(list)
    for problem_name, question in question_entry_list:
        question_lists[problem_name].append(question)

    # Convert per-problem questions into an array
    question_arrays = {k: np.asarray(v) for k, v in question_lists.items()}

    return question_arrays


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
    # Then precedence pair cost uses the term `precedences_inverted[0] - precedences_inverted[1]`.
    res = precedences_inverted[0] - precedences_inverted[1]
    assert m['polarity'] in {'<', '>'}
    if m['polarity'] == '>':
        # If precedence 1 is actually better than precedence 0, we invert the term.
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
