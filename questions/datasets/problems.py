import logging

import tensorflow as tf

from proving import config
from proving import file_path_list
from proving.utils import py_str


def get_datasets_split(patterns, validation_split, max_problems=None):
    problems_all = get_dataset(patterns)
    logging.info('Number of problems available: %d', problems_all.cardinality())
    logging.debug('Leading 10 problems: %s', [py_str(p) for p in problems_all.take(10)])
    if max_problems is not None:
        problems_all = problems_all.take(max_problems)
    logging.info('Number of problems taken: %d', problems_all.cardinality())
    assert 0 <= validation_split <= 1
    problems_validation_count = tf.cast(tf.cast(problems_all.cardinality(), tf.float32) * validation_split, tf.int64)
    problems = {
        'validation': problems_all.take(problems_validation_count),
        'train': problems_all.skip(problems_validation_count)
    }
    for k in problems:
        logging.info(f'Number of {k} problems: %d', problems[k].cardinality())
    return problems


def get_dataset(patterns, shuffle=True, seed=0):
    base_path = config.problems_path()
    patterns_normalized = [file_path_list.normalize_path(pattern, base_path=base_path) for pattern in patterns]
    # filipbartek: From my observation, `list_files` shuffles non-deterministically even if `seed` is specified.
    # We resort to shuffling by `shuffle`, which is deterministic.
    dataset_files = tf.data.Dataset.list_files(patterns_normalized, shuffle=False)
    if shuffle:
        dataset_files = dataset_files.shuffle(dataset_files.cardinality(), seed=seed, reshuffle_each_iteration=False)
    return dataset_files.map(problem_path_to_name, deterministic=True)


def problem_path_to_name(path):
    base_name = tf.strings.split(path, sep='/')[-1]
    problem_name = tf.strings.regex_replace(base_name,
                                            r'^(?P<problem_name>[A-Z]{3}[0-9]{3}[-+^=_][1-9][0-9]*(?:\.[0-9]{3})*)\.p$',
                                            r'\1', False)
    return problem_name
