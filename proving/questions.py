#!/usr/bin/env python3

import argparse
import datetime
import itertools
import logging
import os
import re
import warnings

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
    parser.add_argument('--question-dir', required=True)
    parser.add_argument('--signature-dir', required=True)
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('--test-size', type=float)
    parser.add_argument('--train-size', type=float)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--evaluation-period', type=int, default=1)
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--use-bias', action='store_true')
    parser.add_argument('--random-weights', type=int, default=0)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(threadName)s %(levelname)s - %(message)s')

    np.random.seed(0)
    tf.random.set_seed(0)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(args.log_dir, current_time, 'train')
    test_log_dir = os.path.join(args.log_dir, current_time, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    tf.summary.experimental.set_step(0)

    with joblib.parallel_backend('threading', n_jobs=args.jobs):
        questions = get_problem_questions(args.question_dir)
        problem_names = list(questions.keys())
        logging.info(f'Number of problems: {len(problem_names)}')
        signatures = get_problem_signatures(args.signature_dir, 'predicate', problem_names)
        problems = {problem_name: {'questions': questions[problem_name], 'symbol_embeddings': signatures[problem_name]} for
                    problem_name in problem_names}
        with train_summary_writer.as_default():
            tf.summary.histogram('symbols_per_problem', [len(d['symbol_embeddings']) for d in problems.values()])
            tf.summary.histogram('questions_per_problem', [len(d['questions']) for d in problems.values()])
        problems_list = list(problems.items())

        if args.train_size == 1.0:
            problems_train = problems_list
            problems_test = []
        else:
            problems_train, problems_test = train_test_split(problems_list,
                                                             test_size=args.test_size,
                                                             train_size=args.train_size,
                                                             random_state=0)
        logging.info(f'Number of training problems: {len(problems_train)}')
        logging.info(f'Number of test problems: {len(problems_test)}')

        x_train, sample_weight_train = problems_to_data(problems_train)
        x_test, sample_weight_test = problems_to_data(problems_test)

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
            for w in tqdm(list(itertools.chain(np.eye(k), np.eye(k) * -1,
                                               (np.random.normal(0, 1, k) for _ in range(args.random_weights)))),
                          unit='weight', desc='Evaluating custom weights'):
                if args.use_bias:
                    weights = [w.reshape(-1, 1), np.zeros(1)]
                else:
                    weights = [w.reshape(-1, 1)]
                model = get_model(k, args.use_bias, weights)
                record = evaluate(model, x_test, sample_weight_test, x_train, sample_weight_train, loss_fn)
                records.append(record)

            for iteration_i in tqdm(range(args.iterations), unit='iteration', desc='Training repeatedly'):
                model = get_model(k, args.use_bias)
                if iteration_i == 0:
                    keras.utils.plot_model(model, 'model.png', show_shapes=True)
                with tqdm(range(args.epochs), unit='epoch', desc='Training once') as t:
                    for epoch_i in t:
                        tf.summary.experimental.set_step(epoch_i)
                        if epoch_i % args.evaluation_period == 0:
                            record = {
                                'iteration': iteration_i,
                                'epoch': epoch_i
                            }
                            record.update(evaluate(model, x_test, sample_weight_test, x_train, sample_weight_train, loss_fn,
                                                   test_summary_writer, train_summary_writer))
                            records.append(record)
                        loss_value = train_step(model, x_train, sample_weight_train, loss_fn, optimizer)
                        with train_summary_writer.as_default():
                            tf.summary.scalar('loss', loss_value)
                        t.set_postfix({'loss': loss_value.numpy()})
        finally:
            save_df(
                utils.dataframe_from_records(records, dtypes={'iteration': pd.UInt32Dtype(), 'epoch': pd.UInt32Dtype()}),
                'measurements')


def evaluate(model, x_test, sample_weight_test, x_train, sample_weight_train, loss_fn, test_summary_writer=None,
             train_summary_writer=None):
    record = {}
    weights = model.get_weights()
    for weight_i, weight in enumerate(weights):
        record[('weight', weight_i)] = np.squeeze(weight)
        record[('weight_normalized', weight_i)] = np.squeeze(sklearn.preprocessing.normalize(weight, axis=0))
    if train_summary_writer is not None:
        with train_summary_writer.as_default():
            tf.summary.text('weights', str(weights))
    for name, value in test_step(model, x_test, sample_weight_test, loss_fn).items():
        record[('test', name)] = value.numpy()
        if test_summary_writer is not None:
            with test_summary_writer.as_default():
                tf.summary.scalar(name, value)
    for name, value in test_step(model, x_train, sample_weight_train, loss_fn).items():
        record[('train', name)] = value.numpy()
        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                tf.summary.scalar(name, value)
    return record


def train_step(model, x, sample_weight, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(np.ones((len(sample_weight), 1), dtype=np.bool), logits, sample_weight=sample_weight)
        # loss_value is average loss over samples (questions).
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


def test_step(model, x, sample_weight, loss_fn):
    if x is None:
        return {}
    logits = model(x, training=False)
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('probs', tf.sigmoid(logits))
    res = {'loss': loss_fn(np.ones((len(sample_weight), 1), dtype=np.bool), logits, sample_weight=sample_weight)}
    metrics = {
        'accuracy': keras.metrics.BinaryAccuracy(threshold=0),
        'crossentropy': keras.metrics.BinaryCrossentropy(from_logits=True)
    }
    for name, metric in metrics.items():
        metric.update_state(np.ones((len(sample_weight), 1), dtype=np.bool), logits, sample_weight=sample_weight)
        res[name] = metric.result()
    return res


def problems_to_data(problems):
    if len(problems) == 0:
        return None, None
    x_lists = {'symbol_embeddings': [], 'ranking_difference': [], 'segment_ids': []}
    sample_weight_list = []
    question_i = 0
    for problem_name, d in problems:
        symbol_embeddings = d['symbol_embeddings']
        n = len(symbol_embeddings)
        questions = d['questions']
        assert questions.dtype == np.float32
        m = len(questions)
        x_lists['symbol_embeddings'].append(np.tile(symbol_embeddings, (m, 1)))
        x_lists['ranking_difference'].append(questions.reshape(m * n))
        x_lists['segment_ids'].append(np.repeat(np.arange(question_i, question_i + m, dtype=np.uint32), n))
        sample_weight_list.append(np.full(m, 1 / m, dtype=dtype_tf_float))
        question_i += m
    x = {k: np.concatenate(v) for k, v in x_lists.items()}
    sample_weight = np.concatenate(sample_weight_list)
    return x, sample_weight


def get_model(k, use_bias=False, weights=None):
    symbol_embeddings = keras.Input(shape=k, name='symbol_embeddings')
    symbol_costs_layer = layers.Dense(1, use_bias=use_bias, name='symbol_costs')
    symbol_costs = symbol_costs_layer(symbol_embeddings)
    if weights is not None:
        symbol_costs_layer.set_weights(weights)
    ranking_difference = keras.Input(shape=1, name='ranking_difference')
    potentials = layers.multiply([symbol_costs, ranking_difference])
    segment_ids = keras.Input(shape=1, name='segment_ids', dtype=tf.int32)
    precedence_pair_logit = tf.math.segment_sum(keras.backend.flatten(potentials), keras.backend.flatten(segment_ids))
    precedence_pair_logit = layers.Flatten()(precedence_pair_logit)
    return keras.Model(inputs=[symbol_embeddings, ranking_difference, segment_ids], outputs=precedence_pair_logit)


@memory.cache
def get_problem_questions(question_dir):
    def load_question_dir_entry(dir_entry):
        m = re.search(
            r'^(?P<problem_name>(?P<problem_domain>[A-Z]{3})(?P<problem_number>[0-9]{3})(?P<problem_form>[-+^=_])(?P<problem_version>[1-9])(?P<problem_size_parameters>[0-9]*(\.[0-9]{3})*))_(?P<question_number>\d+)\.q$',
            dir_entry.name, re.MULTILINE)
        problem_name = m['problem_name']
        return problem_name, load_question(dir_entry.path)

    question_list = Parallel(verbose=1)(delayed(load_question_dir_entry)(dir_entry) for dir_entry in os.scandir(question_dir))
    questions = {}
    for problem_name, question in question_list:
        if problem_name not in questions:
            questions[problem_name] = []
        questions[problem_name].append(question)
    for problem_name in questions:
        questions[problem_name] = np.asarray(questions[problem_name])
    return questions


@memory.cache
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
    #logging.debug(f'n={n}, abs.sum={np.sum(np.abs(res))}, abs.std={np.std(np.abs(res))}, std={np.std(res)}')
    return res


def precedence_from_string(s):
    return np.fromstring(s, sep=',', dtype=np.uint32)


if __name__ == '__main__':
    main()
