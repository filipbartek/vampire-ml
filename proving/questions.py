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
import dgl
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed
from ordered_set import OrderedSet
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

from proving import symbols
from proving import utils
from proving.graphifier import Graphifier
from proving.heterographconv import HeteroGCN
from proving.memory import memory
from proving.solver import Solver
from proving.utils import number_of_nodes
from vampire_ml.results import save_df

dtype_tf_float = np.float32

metrics = {
    'accuracy': keras.metrics.BinaryAccuracy(threshold=0),
    'crossentropy': keras.metrics.BinaryCrossentropy(from_logits=True)
}

metric_names = ['loss'] + list(metrics.keys())


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
    parser.add_argument('--evaluate-on-training-set', action='store_true')
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--use-bias', action='store_true')
    parser.add_argument('--hidden-units', type=int, default=0)
    parser.add_argument('--standard-weights', action='store_true')
    parser.add_argument('--random-weights', type=int, default=0)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--max-data-length', type=int, default=100000)
    parser.add_argument('--max-problems', type=int)
    parser.add_argument('--max-questions-per-problem', type=int)
    parser.add_argument('--max-problem-size', type=int)
    parser.add_argument('--max-batch-size', type=int)
    parser.add_argument('--log-level', default='INFO', choices=['INFO', 'DEBUG'])
    parser.add_argument('--plot-model')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format='%(asctime)s %(threadName)s %(levelname)s - %(message)s')

    np.random.seed(0)
    tf.random.set_seed(0)

    logging.info(f'Cache location: {memory.location}')
    logging.debug('TensorFlow physical devices: %s', tf.config.experimental.list_physical_devices())

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    subset_names = ('train', 'test')
    logging.info(f'Log directory: {os.path.join(args.log_dir, current_time)}')
    summary_writers = {name: tf.summary.create_file_writer(os.path.join(args.log_dir, current_time, name)) for name in
                       subset_names}
    tf.summary.experimental.set_step(0)

    with summary_writers['train'].as_default():
        tf.summary.text('args', str(args))

    with joblib.parallel_backend('threading', n_jobs=args.jobs):
        batch_generator = BatchGenerator(args.max_data_length)
        solver = Solver(timeout=20)
        graphifier = Graphifier(solver, max_number_of_nodes=args.max_problem_size)
        if args.evaluate_on_training_set:
            eval_dataset_names = ('test', 'train')
        else:
            eval_dataset_names = ('test',)
        problems, eval_datasets = get_data(args.question_dir, graphifier, args.cache_file, args.train_size,
                                           args.test_size, batch_generator, max_problems=args.max_problems,
                                           max_questions_per_problem=args.max_questions_per_problem,
                                           output_dir=args.output, datasets=eval_dataset_names)

        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

        optimizers = {
            'sgd': keras.optimizers.SGD,
            'adam': keras.optimizers.Adam,
            'rmsprop': keras.optimizers.RMSprop
        }
        optimizer = optimizers[args.optimizer](learning_rate=args.learning_rate)

        records = []

        if args.profile:
            tf.profiler.experimental.start(args.log_dir)
        try:
            model = SymbolPreferenceGCN('predicate', graphifier.canonical_etypes, graphifier.ntypes)
            if args.plot_model is not None:
                keras.utils.plot_model(model, utils.path_join(args.output, args.plot_model, makedir=True),
                                       show_shapes=True)
            rng = np.random.RandomState(0)
            if args.max_batch_size is None:
                batch_generator_train = batch_generator
            else:
                batch_generator_train = BatchGenerator(min(args.max_batch_size, args.max_data_length))
            with tqdm(range(args.epochs), unit='epoch', desc='Training') as t:
                for i in t:
                    tf.summary.experimental.set_step(i)
                    if i % args.evaluation_period == 0:
                        with tf.profiler.experimental.Trace('test', step_num=i, _r=1):
                            record = {'type': 'training', 'step': i}
                            record.update(evaluate(model, eval_datasets, loss_fn, summary_writers))
                            records.append(record)
                    with tf.profiler.experimental.Trace('train', step_num=i, _r=1):
                        with summary_writers['train'].as_default():
                            loss_value = train_step(model, problems['train'], loss_fn, optimizer, rng,
                                                    batch_generator_train)
                        t.set_postfix({'loss': loss_value.numpy()})
            i = args.epochs
            tf.summary.experimental.set_step(i)
            record = {'type': 'training', 'step': i}
            record.update(evaluate(model, eval_datasets, loss_fn, summary_writers))
            records.append(record)
        finally:
            if args.profile:
                tf.profiler.experimental.stop()
            save_df(utils.dataframe_from_records(records, dtypes={'type': 'category', 'step': pd.UInt32Dtype()}),
                    'evaluation', args.output)


class SymbolPreferenceGCN(keras.Model):
    def __init__(self, symbol_type, canonical_etypes, ntypes, num_layers=4):
        super().__init__()
        self.symbol_type = symbol_type
        edge_layer_sizes = {canonical_etype: 64 for canonical_etype in canonical_etypes}
        node_layer_sizes = {ntype: 64 for ntype in ntypes}
        self.hetero_gcn = HeteroGCN(edge_layer_sizes, node_layer_sizes, num_layers, False, [symbol_type])
        self.cost_model = layers.Dense(1)

    def call(self, x):
        graph = x['batch_graph']
        # Row: problem -> symbol
        symbol_embeddings = self.hetero_gcn(graph)[self.symbol_type]
        symbol_costs = tf.squeeze(self.cost_model(symbol_embeddings))
        # Row: problem -> question -> symbol
        question_symbols = x['question_symbols']
        symbol_costs_tiled = tf.gather(symbol_costs, question_symbols)
        ranking_difference = x['ranking_difference']
        potentials = tf.multiply(symbol_costs_tiled, ranking_difference)
        segment_ids = x['segment_ids']
        # Row: problem -> question
        precedence_pair_logit = tf.math.segment_sum(potentials, segment_ids)
        return precedence_pair_logit


@memory.cache(ignore=['cache_file', 'output_dir'], verbose=2)
def get_data(question_dir, graphifier, cache_file, train_size, test_size, batch_generator, max_problems,
             max_questions_per_problem, random_state=0, output_dir=None, datasets=None):
    problems_all = get_problems(question_dir, graphifier, max_problems, max_questions_per_problem,
                                np.random.RandomState(random_state), cache_file, output_dir)
    logging.info(f'Number of problems graphified: {len(problems_all)}')

    if train_size == 1.0 or test_size == 0.0:
        problems = {
            'train': list(problems_all.values()),
            'test': []
        }
    elif test_size == 1.0 or train_size == 0.0:
        problems = {
            'test': list(problems_all.values()),
            'train': []
        }
    else:
        problems_train, problems_test = train_test_split(list(problems_all.keys()),
                                                         test_size=test_size,
                                                         train_size=train_size,
                                                         random_state=random_state)
        problems = {
            'train': [problems_all[k] for k in problems_train],
            'test': [problems_all[k] for k in problems_test]
        }
    logging.info(f'Number of training problems: %d', len(problems['train']))
    logging.info(f'Number of test problems: %d', len(problems['test']))
    if datasets is None:
        datasets = ('test', 'train')
    data = {name: batch_generator.get_batches(problems[name]) for name in datasets}
    return problems, data


@memory.cache(ignore=['cache_file', 'output_dir'], verbose=2)
def get_problems(question_dir, graphifier, max_problems, max_questions_per_problem, rng, cache_file, output_dir):
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
    graphs_records = graphifier.problems_to_graphs(problem_names)
    graphs, records = zip(*graphs_records)
    if output_dir is not None:
        save_df(utils.dataframe_from_records(records), 'graphs', output_dir)
    problems = {problem_name: {'graph': graphs[i], 'questions': questions[problem_name]} for
                i, problem_name in enumerate(problem_names) if graphs[i] is not None}
    if cache_file is not None:
        logging.info(f'Saving problems into {cache_file}...')
        pickle.dump(problems, open(cache_file, mode='wb'))
        logging.info(f'Problems saved into {cache_file}.')
    return problems


def evaluate(model, datasets, loss_fn, summary_writers=None):
    record = {}
    for dataset_name, dataset in datasets.items():
        for name, value in test_step(model, dataset, loss_fn).items():
            record[(dataset_name, name)] = value.numpy()
            if summary_writers is not None:
                with summary_writers[dataset_name].as_default():
                    tf.summary.scalar(name, value)
    return record


def train_step(model, problems, loss_fn, optimizer, rng, batch_generator):
    x, sample_weight = batch_generator.get_batch_random(problems, rng)
    tf.summary.scalar('batch.problems', x['batch_graph'].batch_size)
    tf.summary.scalar('batch.ranking_difference.len', len(x['ranking_difference']))
    tf.summary.scalar('batch.sample_weight.len', len(sample_weight))
    tf.summary.scalar('batch.sample_weight.sum', np.sum(sample_weight))
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        assert len(logits) == len(sample_weight)
        loss_value = loss_fn(np.ones((len(sample_weight), 1), dtype=np.bool), tf.expand_dims(logits, 1),
                             sample_weight=sample_weight)
        # loss_value is average loss over samples (questions).
    tf.summary.scalar('batch.loss', loss_value)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


def test_step(model, data, loss_fn):
    if data is None:
        return {}
    xs = data['xs']
    sample_weight = data['sample_weight']
    logits = np.concatenate(
        [model(x, training=False).numpy() for x in tqdm(xs, unit='batch', desc='Evaluating', disable=len(xs) < 10)])
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('probs', tf.sigmoid(logits))
    assert len(logits) == len(sample_weight)
    # BinaryCrossentropy requires that there is at least one non-batch dimension.
    res = {'loss': loss_fn(np.ones((len(sample_weight), 1), dtype=np.bool), tf.expand_dims(logits, 1),
                           sample_weight=sample_weight)}
    for name, metric in metrics.items():
        metric.update_state(np.ones((len(sample_weight), 1), dtype=np.bool), tf.expand_dims(logits, 1),
                            sample_weight=sample_weight)
        res[name] = metric.result()
        metric.reset_states()
    return res


class BatchGenerator:
    def __init__(self, max_batch_length):
        self.max_batch_length = max_batch_length

    @classmethod
    def problem_size(cls, d, n_questions=None):
        g = d['graph']
        if n_questions is None:
            n_questions = len(d['questions'])
        assert n_questions <= len(d['questions'])
        return number_of_nodes(g) + d['questions'].shape[1] * n_questions

    def get_batches(self, problems):
        if len(problems) == 0:
            # This happens namely when the user sets test set size to 0.
            return None
        xs = []
        sample_weight_list = []
        problem_sizes = dict(enumerate(map(self.problem_size, problems)))
        max_size = max(problem_sizes.values())
        logging.info(f'Maximum problem size: {max_size}')
        if max_size > self.max_batch_length:
            raise RuntimeError(
                f'Failed to distribute problems in batches of size at most {self.max_batch_length}. Problem of size {max_size} encountered.')
        bins = binpacking.to_constant_volume(problem_sizes, self.max_batch_length)
        logging.info(f'Number of batches: {len(bins)}')
        for cur_bin in bins:
            batch_size = sum(cur_bin.values())
            assert batch_size <= self.max_batch_length
            x, sample_weight = self._get_batch(problems[i] for i in cur_bin.keys())
            xs.append(x)
            sample_weight_list.append(sample_weight)
        sample_weight = np.concatenate(sample_weight_list)
        return {'xs': xs, 'sample_weight': sample_weight}

    def get_batch_random(self, problems, rng):
        question_ids = list(itertools.chain.from_iterable(
            ((problem_i, question_i) for question_i in range(len(problems[problem_i]['questions']))) for problem_i in
            range(len(problems))))
        perm = rng.permutation(question_ids)
        selected_ids = collections.defaultdict(list)
        total_size = 0
        for problem_i, question_i in perm:
            d = problems[problem_i]
            if problem_i in selected_ids:
                cur_graph_size = 0
            else:
                cur_graph_size = number_of_nodes(d['graph'])
            cur_size = cur_graph_size + d['questions'].shape[1]
            if cur_size > self.max_batch_length:
                raise RuntimeError(f'A question of size {cur_size} does not fit into a batch.')
            if total_size + cur_size > self.max_batch_length:
                break
            selected_ids[problem_i].append(question_i)
            total_size += cur_size
        return self._get_batch((problems[i] for i in selected_ids), selected_ids.values())

    def _get_batch(self, problems, question_ids=None):
        """Does not ensure that the batch is sufficiently small."""
        graphs = []
        x_lists = {'ranking_difference': [], 'question_symbols': [], 'segment_ids': []}
        sample_weight_list = []
        question_i = 0
        symbol_i = 0
        if question_ids is None:
            question_ids = itertools.repeat(None)
        total_size = 0
        for d, qids in zip(problems, question_ids):
            graphs.append(d['graph'])
            if qids is None:
                # We take all the questions for this problem.
                questions = d['questions']
            else:
                questions = d['questions'][qids]
            assert questions.dtype == np.float32
            m, n = questions.shape
            x_lists['ranking_difference'].append(questions.flatten())
            x_lists['question_symbols'].append(np.tile(np.arange(symbol_i, symbol_i + n, dtype=np.int32), m).flatten())
            x_lists['segment_ids'].append(np.repeat(np.arange(question_i, question_i + m, dtype=np.int32), n).flatten())
            sample_weight_list.append(np.full(m, 1 / m, dtype=dtype_tf_float))
            question_i += m
            symbol_i += n
            total_size += self.problem_size(d, len(questions))
        assert total_size <= self.max_batch_length
        x = {k: np.concatenate(v) for k, v in x_lists.items()}
        assert len(x['ranking_difference']) == len(x['question_symbols']) == len(x['segment_ids'])
        assert symbol_i == np.max(x['question_symbols']) + 1
        sample_weight = np.concatenate(sample_weight_list)
        assert len(sample_weight) == question_i
        x['batch_graph'] = dgl.batch(graphs)
        logging.debug(
            'Batch created. Size: %d/%d (actual/maximum). Problems: %d. Nodes: %d. Questions: %d. Cumulative symbol count: %d. Cumulative question length: %d.',
            total_size, self.max_batch_length, len(graphs), number_of_nodes(x['batch_graph']), question_i, symbol_i,
            len(x['ranking_difference']))
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
def get_problem_questions(question_dir, rng, max_problems=None, max_questions_per_problem=None):
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

    # Ensure we have at most max_problems problems
    problem_names = OrderedSet(tuple(zip(*question_entry_list))[0])
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
