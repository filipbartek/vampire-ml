#!/usr/bin/env python3

import argparse
import collections
import datetime
import itertools
import logging
import os
import pickle
import re
import warnings

import binpacking
import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

from proving import symbols
from proving import utils
from proving.memory import memory
from vampire_ml.results import save_df

dtype_tf_float = np.float32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('--question-dir')
    parser.add_argument('--signature-dir')
    parser.add_argument('--cache-file')
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('--test-size', type=float)
    parser.add_argument('--train-size', type=float)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--evaluation-period', type=int, default=1)
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--use-bias', action='store_true')
    parser.add_argument('--hidden-units', type=int, default=0)
    parser.add_argument('--standard-weights', action='store_true')
    parser.add_argument('--random-weights', type=int, default=0)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--tf-log-device-placement', action='store_true')
    parser.add_argument('--max-data-length', type=int, default=64 * 1024 * 1024)
    parser.add_argument('--max-questions-per-problem', type=int)
    parser.add_argument('--log-level', default='INFO', choices=['INFO', 'DEBUG'])
    parser.add_argument('--plot-model')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format='%(asctime)s %(threadName)s %(levelname)s - %(message)s')
    tf.debugging.set_log_device_placement(args.tf_log_device_placement)

    np.random.seed(0)
    tf.random.set_seed(0)

    logging.info(f'Cache location: {memory.location}')
    logging.debug('TensorFlow physical devices: %s', tf.config.experimental.list_physical_devices())

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(args.log_dir, current_time, 'train')
    test_log_dir = os.path.join(args.log_dir, current_time, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    tf.summary.experimental.set_step(0)

    with train_summary_writer.as_default():
        tf.summary.text('args', str(args))

    with joblib.parallel_backend('threading', n_jobs=args.jobs):
        batch_generator = BatchGenerator(args.max_data_length)
        problems_train, problems_test, data_train, data_test = get_data(args.question_dir, args.signature_dir,
                                                                        args.cache_file, args.train_size,
                                                                        args.test_size, batch_generator,
                                                                        args.max_questions_per_problem)

        k = 12

        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

        optimizers = {
            'sgd': keras.optimizers.SGD,
            'adam': keras.optimizers.Adam,
            'rmsprop': keras.optimizers.RMSprop
        }
        optimizer = optimizers[args.optimizer](learning_rate=args.learning_rate)

        records = []

        try:
            w_values_list = []
            if args.standard_weights:
                w_values_list.extend((np.eye(k), np.eye(k) * -1))
            if args.random_weights > 0:
                rng = np.random.RandomState(0)
                w_values_list.append(rng.normal(0, 1, (args.random_weights, k)))
            if len(w_values_list) > 0:
                for w in tqdm(np.concatenate(w_values_list), unit='weight', desc='Evaluating custom weights'):
                    model = get_model(k, weights=[w.reshape(-1, 1)])
                    record = evaluate(model, data_test, data_train, loss_fn, extract_weights=True)
                    records.append(record)

            model = get_model(k, use_bias=args.use_bias, hidden_units=args.hidden_units)
            if args.plot_model is not None:
                p = args.plot_model
                if args.output is not None:
                    p = os.path.join(args.output, p)
                keras.utils.plot_model(model, p, show_shapes=True)
            rng = np.random.RandomState(0)
            with tqdm(range(args.epochs), unit='epoch', desc='Training') as t:
                for epoch_i in t:
                    tf.summary.experimental.set_step(epoch_i)
                    if epoch_i % args.evaluation_period == 0:
                        record = {'epoch': epoch_i}
                        record.update(evaluate(model, data_test, data_train, loss_fn,
                                               test_summary_writer, train_summary_writer,
                                               extract_weights=(args.hidden_units == 0)))
                        records.append(record)
                    with train_summary_writer.as_default():
                        loss_value = train_step(model, problems_train, loss_fn, optimizer, rng, batch_generator)
                    t.set_postfix({'loss': loss_value.numpy()})
                epoch_i = args.epochs
                tf.summary.experimental.set_step(epoch_i)
                record = {'epoch': epoch_i}
                record.update(evaluate(model, data_test, data_train, loss_fn,
                                       test_summary_writer, train_summary_writer,
                                       extract_weights=(args.hidden_units == 0)))
                records.append(record)
        finally:
            save_df(utils.dataframe_from_records(records, dtypes={'epoch': pd.UInt32Dtype()}), 'evaluation', args.output)


@memory.cache(ignore=['cache_file'], verbose=2)
def get_data(question_dir, signature_dir, cache_file, train_size, test_size, batch_generator, max_questions_per_problem,
             random_state=0):
    problems = get_problems(question_dir, signature_dir, max_questions_per_problem, cache_file)
    logging.info(f'Number of problems: {len(problems)}')
    if train_size == 1.0:
        problems_train = problems
        problems_test = []
    else:
        problems_train, problems_test = train_test_split(problems,
                                                         test_size=test_size,
                                                         train_size=train_size,
                                                         random_state=random_state)
    logging.info(f'Number of training problems: {len(problems_train)}')
    logging.info(f'Number of test problems: {len(problems_test)}')
    data_train = batch_generator.get_batches(problems_train)
    data_test = batch_generator.get_batches(problems_test)
    return problems_train, problems_test, data_train, data_test


@memory.cache(ignore=['cache_file'], verbose=2)
def get_problems(question_dir, signature_dir, max_questions_per_problem, cache_file):
    if cache_file is not None:
        try:
            logging.info(f'Loading problems from {cache_file}...')
            problems = pickle.load(open(cache_file, mode='rb'))
            logging.info(f'Problems loaded from {cache_file}.')
            return problems
        except FileNotFoundError:
            pass
    questions = get_problem_questions(question_dir, max_questions_per_problem)
    problem_names = list(questions.keys())
    signatures = get_problem_signatures(signature_dir, 'predicate', problem_names)
    problems = [{'questions': questions[problem_name], 'symbol_embeddings': signatures[problem_name]} for problem_name
                in problem_names]
    if cache_file is not None:
        logging.info(f'Saving problems into {cache_file}...')
        pickle.dump(problems, open(cache_file, mode='wb'))
        logging.info(f'Problems saved into {cache_file}.')
    return problems


def evaluate(model, data_test, data_train, loss_fn, test_summary_writer=None, train_summary_writer=None,
             extract_weights=False):
    record = {}
    if extract_weights:
        weights = model.get_layer('symbol_costs').get_weights()
        for weight_i, weight in enumerate(weights):
            record[('weight', weight_i)] = np.squeeze(weight)
            record[('weight_normalized', weight_i)] = np.squeeze(sklearn.preprocessing.normalize(weight, axis=0))
        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                tf.summary.text('weights', str(weights))
    for name, value in test_step(model, data_test, loss_fn).items():
        record[('test', name)] = value.numpy()
        if test_summary_writer is not None:
            with test_summary_writer.as_default():
                tf.summary.scalar(name, value)
    for name, value in test_step(model, data_train, loss_fn).items():
        record[('train', name)] = value.numpy()
        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                tf.summary.scalar(name, value)
    return record


def train_step(model, problems, loss_fn, optimizer, rng, batch_generator):
    problems_selected = (problems[i] for i in rng.permutation(len(problems)))
    x, sample_weight = batch_generator.get_batch(problems_selected)
    tf.summary.scalar('batch.symbol_embeddings.len', len(x['symbol_embeddings']))
    tf.summary.scalar('batch.ranking_difference.len', len(x['ranking_difference']))
    tf.summary.scalar('batch.sample_weight.len', len(sample_weight))
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        assert len(logits) == len(sample_weight)
        loss_value = loss_fn(np.ones((len(sample_weight), 1), dtype=np.bool), logits, sample_weight=sample_weight)
        # loss_value is average loss over samples (questions).
    tf.summary.scalar('batch.loss', loss_value)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


def test_step(model, data, loss_fn):
    if data is None:
        return {}
    xs, sample_weight = data
    logits = np.concatenate([model(x, training=False).numpy() for x in xs])
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('probs', tf.sigmoid(logits))
    assert len(logits) == len(sample_weight)
    res = {'loss': loss_fn(np.ones((len(sample_weight), 1), dtype=np.bool), logits, sample_weight=sample_weight)}
    metrics = {
        'accuracy': keras.metrics.BinaryAccuracy(threshold=0),
        'crossentropy': keras.metrics.BinaryCrossentropy(from_logits=True)
    }
    for name, metric in metrics.items():
        metric.update_state(np.ones((len(sample_weight), 1), dtype=np.bool), logits, sample_weight=sample_weight)
        res[name] = metric.result()
    return res


class BatchGenerator:
    def __init__(self, max_batch_length):
        self.max_batch_length = max_batch_length

    def get_batches(self, problems):
        if len(problems) == 0:
            # This happens namely when the user sets test set size to 0.
            return None
        xs = []
        sample_weight_list = []
        problem_sizes = {i: len(d['symbol_embeddings']) * len(d['questions']) for i, d in enumerate(problems)}
        bins = binpacking.to_constant_volume(problem_sizes, self.max_batch_length)
        for cur_bin in bins:
            batch_size = sum(cur_bin.values())
            if batch_size > self.max_batch_length:
                assert len(cur_bin) == 1
                raise RuntimeError(
                    f'Failed to distribute problems in batches of size at most {self.max_batch_length}. Problem of size {batch_size} encountered.')
            x, sample_weight = self.get_batch(problems[i] for i in cur_bin.keys())
            xs.append(x)
            sample_weight_list.append(sample_weight)
        logging.info(f'Number of batches: {len(xs)}')
        sample_weight = np.concatenate(sample_weight_list)
        logging.debug(f'Sample weight: Shape: {sample_weight.shape}. Sizes in bytes: {sample_weight.nbytes}.')
        return xs, sample_weight

    def get_batch(self, problems):
        x_lists = {'symbol_embeddings': [], 'ranking_difference': [], 'question_symbols': [], 'segment_ids': []}
        sample_weight_list = []
        question_i = 0
        symbol_i = 0
        for d in problems:
            symbol_embeddings = d['symbol_embeddings']
            n = len(symbol_embeddings)
            questions = d['questions']
            assert questions.dtype == np.float32
            m = len(questions)
            if self.max_batch_length is not None and len(x_lists['ranking_difference']) + m * n > self.max_batch_length:
                break
            x_lists['symbol_embeddings'].append(symbol_embeddings)
            x_lists['ranking_difference'].append(questions.reshape(m * n, 1))
            x_lists['question_symbols'].append(
                np.tile(np.arange(symbol_i, symbol_i + n, dtype=np.int32), m).reshape(m * n, 1))
            x_lists['segment_ids'].append(
                np.repeat(np.arange(question_i, question_i + m, dtype=np.int32), n).reshape(m * n, 1))
            sample_weight_list.append(np.full(m, 1 / m, dtype=dtype_tf_float))
            question_i += m
            symbol_i += n
        x = {k: np.concatenate(v) for k, v in x_lists.items()}
        assert len(x['ranking_difference']) == len(x['question_symbols']) == len(x['segment_ids'])
        assert symbol_i == len(x['symbol_embeddings']) == np.max(x['question_symbols']) + 1
        logging.debug(f'Created batch. Total size in bytes: %d. Shapes: %s. Sizes in bytes: %s.',
                      sum(v.nbytes for v in x.values()),
                      {k: v.shape for k, v in x.items()},
                      {k: v.nbytes for k, v in x.items()})
        sample_weight = np.concatenate(sample_weight_list)
        logging.debug(f'Sample weight: Shape: {sample_weight.shape}. Sizes in bytes: {sample_weight.nbytes}.')
        return x, sample_weight


def get_model(k, weights=None, use_bias=False, hidden_units=0):
    # Row: problem -> symbol
    symbol_embeddings = keras.Input(shape=k, name='symbol_embeddings')
    x = symbol_embeddings
    if hidden_units > 0:
        x = layers.Dense(hidden_units, 'relu')(x)
    symbol_costs_layer = layers.Dense(1, use_bias=use_bias, name='symbol_costs')
    symbol_costs = symbol_costs_layer(x)
    if weights is not None:
        symbol_costs_layer.set_weights(weights)
    # Row: problem -> question -> symbol
    ranking_difference = keras.Input(shape=1, name='ranking_difference')
    question_symbols = keras.Input(shape=1, name='question_symbols', dtype=tf.int32)
    symbol_costs_tiled = layers.Flatten(name='symbol_costs_tiled')(tf.gather(symbol_costs, question_symbols))
    potentials = layers.multiply([symbol_costs_tiled, ranking_difference])
    segment_ids = keras.Input(shape=1, name='segment_ids', dtype=tf.int32)
    precedence_pair_logit = tf.math.segment_sum(potentials, keras.backend.flatten(segment_ids))
    precedence_pair_logit = layers.Flatten(name='logits')(precedence_pair_logit)
    return keras.Model(inputs=[symbol_embeddings, ranking_difference, question_symbols, segment_ids],
                       outputs=precedence_pair_logit)


@memory.cache(verbose=2)
def get_problem_questions(question_dir, max_questions_per_problem):
    def parse_question_dir_entry(dir_entry):
        m = re.search(
            r'^(?P<problem_name>(?P<problem_domain>[A-Z]{3})(?P<problem_number>[0-9]{3})(?P<problem_form>[-+^=_])(?P<problem_version>[1-9])(?P<problem_size_parameters>[0-9]*(\.[0-9]{3})*))_(?P<question_number>\d+)\.q$',
            dir_entry.name, re.MULTILINE)
        problem_name = m['problem_name']
        return problem_name, dir_entry.path

    # Parse paths
    logging.info(f'Parsing question paths in directory {question_dir}...')
    question_entry_list = Parallel(verbose=1)(
        delayed(parse_question_dir_entry)(dir_entry) for dir_entry in os.scandir(question_dir))
    logging.info(f'Question paths parsed. Number of questions: {len(question_entry_list)}')

    # Filter problems with too many questions
    question_paths = collections.defaultdict(list)
    for problem_name, question_path in question_entry_list:
        if max_questions_per_problem is None or len(question_paths[problem_name]) < max_questions_per_problem:
            question_paths[problem_name].append(question_path)

    # Load questions
    queries = itertools.chain.from_iterable(question_paths.values())
    logging.info('Loading questions...')
    question_list = Parallel(verbose=1)(delayed(load_question)(question_path) for question_path in queries)
    logging.info(f'Questions loaded. Number of questions: {len(question_list)}')

    # Collect questions into a dictionary
    problem_names = itertools.chain.from_iterable(
        itertools.repeat(problem_name, len(vv)) for problem_name, vv in question_paths.items())
    questions = collections.defaultdict(list)
    for problem_name, question in zip(problem_names, question_list):
        questions[problem_name].append(question)

    # Convert per-problem questions into an array
    for problem_name in questions:
        questions[problem_name] = np.asarray(questions[problem_name])

    return questions


@memory.cache(verbose=2)
def get_problem_signatures(symbols_dir_path, symbol_type, problems=None):
    if problems is not None:
        iterable = ((problem_name, os.path.join(symbols_dir_path, f'{problem_name}.sig')) for problem_name in problems)
        total = len(problems)
    else:
        iterable = ((os.path.splitext(dir_entry.name)[0], dir_entry.path) for dir_entry in os.scandir(symbols_dir_path))
        total = None
    signatures = {}
    with tqdm(iterable, unit='problem', desc='Loading signatures', total=total) as t:
        for problem_name, signature_path in t:
            t.set_postfix_str(signature_path)
            try:
                sym_all = symbols.load(signature_path)
                sym_selected = symbols.symbols_of_type(sym_all, symbol_type)
                signatures[problem_name] = sym_selected.drop('name', axis='columns').astype(dtype_tf_float).values
            except ValueError:
                warnings.warn(f'Failed to load signature: {signature_path}')
    return signatures


def load_question(question_path):
    content = open(question_path).read()
    m = re.search(r'^(?P<precedence_0>[0-9,]+)\n(?P<precedence_1>[0-9,]+)\n(?P<polarity>[<>])$', content, re.MULTILINE)
    assert m['polarity'] in {'<', '>'}
    if m['polarity'] == '<':
        p = (precedence_from_string(m['precedence_0']), precedence_from_string(m['precedence_1']))
    else:
        p = (precedence_from_string(m['precedence_1']), precedence_from_string(m['precedence_0']))
    n = len(p[0])
    assert p[0].shape == p[1].shape == (n,)
    p_inv = (utils.invert_permutation(p[0]), utils.invert_permutation(p[1]))
    res = (p_inv[1].astype(np.int32) - p_inv[0].astype(np.int32)) * 2 / (n * (n + 1))
    res = res.astype(dtype_tf_float)
    assert np.isclose(0, res.sum(), atol=1e-06)
    # logging.debug(f'n={n}, abs.sum={np.sum(np.abs(res))}, abs.std={np.std(np.abs(res))}, std={np.std(res)}')
    return res


def precedence_from_string(s):
    return np.fromstring(s, sep=',', dtype=np.uint32)


if __name__ == '__main__':
    main()
