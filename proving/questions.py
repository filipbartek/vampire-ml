#!/usr/bin/env python3

import argparse
import datetime
import glob
import logging
import os
import re

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

from proving import symbols
from proving import utils
from proving.memory import memory


def main():
    rng = np.random.RandomState(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--question-pattern', required=True)
    parser.add_argument('--signature-dir', required=True)
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('--test-size', type=float)
    parser.add_argument('--train-size', type=float)
    parser.add_argument('--batches', type=int, default=10)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--evaluation-period', type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(threadName)s %(levelname)s - %(message)s')

    question_paths = glob.iglob(args.question_pattern, recursive=True)
    question_paths = list(question_paths)
    problems = get_problem_questions(question_paths, args.signature_dir, 'predicate')
    print(len(problems))
    problems_list = list(problems.items())

    k = 12

    model = get_model(k)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    # model.compile(loss=loss_fn)
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)

    # model.summary()
    keras.utils.plot_model(model, 'model.svg', show_shapes=True)

    if args.train_size == 1.0:
        problems_train = problems_list
        problems_test = []
    else:
        problems_train, problems_test = train_test_split(problems_list,
                                                         test_size=args.test_size,
                                                         train_size=args.train_size,
                                                         random_state=0)
    print(f'Total problems: {len(problems_list)}. Training problems: {len(problems_train)}. Test problems: {len(problems_test)}.')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(args.log_dir, current_time, 'train')
    test_log_dir = os.path.join(args.log_dir, current_time, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for i in tqdm(range(args.batches), unit='batch', desc='Training'):
        if i % args.evaluation_period == 0:
            with test_summary_writer.as_default():
                for name, value in evaluate(model, problems_test).items():
                    tf.summary.scalar(name, value, step=i)
            with train_summary_writer.as_default():
                for name, value in evaluate(model, problems_train).items():
                    tf.summary.scalar(name, value, step=i)
        if args.batch_size is None:
            batch = problems_train
        else:
            batch = [problems_train[j] for j in rng.choice(len(problems_train), args.batch_size, replace=False)]
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_on_problems(model, batch, loss_fn, optimizer), step=i)


def train_on_problems(model, problems, loss_fn, optimizer):
    # https://stackoverflow.com/a/62683800/4054250
    total_loss = 0
    accum_gradient = [tf.zeros_like(this_var) for this_var in model.trainable_variables]
    for problem_name, d in tqdm(problems, unit='problem', desc='Training on a batch of problems', disable=len(problems) < 1000):
        symbol_embeddings = d['symbol_embeddings']
        questions = d['questions']
        m = len(questions)
        with tf.GradientTape() as tape:
            logits = model({'symbol_embeddings': np.tile(symbol_embeddings, (m, 1, 1)),
                            'ranking_difference': questions}, training=True)
            loss_value = loss_fn(np.ones((m, 1)), logits)
            # loss_value is average loss over samples (questions).
        total_loss += loss_value
        grads = tape.gradient(loss_value, model.trainable_weights)
        accum_gradient = [(acum_grad+grad) for acum_grad, grad in zip(accum_gradient, grads)]
    accum_gradient = [this_grad / len(problems) for this_grad in accum_gradient]
    optimizer.apply_gradients(zip(accum_gradient, model.trainable_weights))
    return total_loss / len(problems)


def evaluate(model, problems):
    metrics = {
        'accuracy': keras.metrics.BinaryAccuracy(threshold=0),
        'crossentropy': keras.metrics.BinaryCrossentropy(from_logits=True)
    }
    for problem_name, d in problems:
        symbol_embeddings = d['symbol_embeddings']
        n = symbol_embeddings.shape[0]
        questions = d['questions']
        m = len(questions)
        assert questions.shape == (m, n)
        logits = model({'symbol_embeddings': np.tile(symbol_embeddings, (m, 1, 1)),
                        'ranking_difference': questions}, training=True)
        for metric in metrics.values():
            metric.update_state(np.ones((m, 1)), logits, sample_weight=1 / m)
    return {k: v.result().numpy() for k, v in metrics.items()}


dtype_tf_float = np.float32


def get_model(k):
    symbol_embeddings = keras.Input(shape=(None, k), name='symbol_embeddings')
    symbol_costs = layers.Dense(1, name='symbol_costs')(symbol_embeddings)
    symbol_costs = layers.Flatten()(symbol_costs)
    ranking_difference = keras.Input(shape=(None,), name='ranking_difference')
    precedence_pair_logit = layers.Dot(axes=1)([symbol_costs, ranking_difference])
    return keras.Model(inputs=[symbol_embeddings, ranking_difference], outputs=precedence_pair_logit)


@memory.cache
def get_problem_questions(question_paths, symbols_dir_path, symbol_type):
    problems = {}
    for question_path in tqdm(question_paths, unit='question', desc='Loading questions'):
        question_file = os.path.basename(question_path)
        m = re.search(
            r'^(?P<problem_name>(?P<problem_domain>[A-Z]{3})(?P<problem_number>[0-9]{3})(?P<problem_form>[-+^=_])(?P<problem_version>[1-9])(?P<problem_size_parameters>[0-9]*(\.[0-9]{3})*))_(?P<question_number>\d+)\.q$',
            question_file, re.MULTILINE)
        problem_name = m['problem_name']
        if problem_name not in problems:
            sym_all = symbols.load(os.path.join(symbols_dir_path, f'{problem_name}.sig'))
            sym_selected = symbols.symbols_of_type(sym_all, symbol_type)
            problems[problem_name] = {
                'symbol_embeddings': sym_selected.drop('name', axis='columns').astype(dtype_tf_float).values,
                'questions': []
            }
        problems[problem_name]['questions'].append(load_question(question_path))
    for problem_name in problems:
        problems[problem_name]['questions'] = np.asarray(problems[problem_name]['questions'])
    return problems


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
    res = (p_inv[1].astype(dtype_tf_float) - p_inv[0]) * 2 / (n * (n + 1))
    assert np.isclose(res.sum(), 0)
    logging.debug(f'n={n}, abs.sum={np.sum(np.abs(res))}, abs.std={np.std(np.abs(res))}, std={np.std(res)}')
    return res


def precedence_from_string(s):
    return np.fromstring(s, sep=',', dtype=np.uint32)


if __name__ == '__main__':
    main()
