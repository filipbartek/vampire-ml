#!/usr/bin/env python3

import argparse
import collections
import datetime
import itertools
import logging
import os
import time

import binpacking
import dgl
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

from proving import simple_features
from proving import utils
from proving.graphifier import Graphifier
from proving.heterographconv import HeteroGCN
from proving.load_questions import get_problems
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
    parser.add_argument('--experiment-id')
    parser.add_argument('--test-size', type=float)
    parser.add_argument('--train-size', type=float)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--evaluation-period', type=int, default=1)
    parser.add_argument('--evaluate-on-training-set', action='store_true')
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--max-test-batch-size', type=int, default=10000000)
    parser.add_argument('--max-train-batch-size', type=int, default=100000)
    parser.add_argument('--train-batch-problems', type=int)
    parser.add_argument('--max-problems', type=int)
    parser.add_argument('--max-problem-size', type=int)
    parser.add_argument('--max-questions-per-problem', type=int)
    parser.add_argument('--log-level', default='INFO', choices=['INFO', 'DEBUG'])
    parser.add_argument('--plot-model')
    parser.add_argument('--profile-start', type=int)
    parser.add_argument('--profile-stop', type=int)
    parser.add_argument('--device')
    parser.add_argument('--evaluate-linear-standard', action='store_true')
    parser.add_argument('--evaluate-linear-random', type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format='%(asctime)s %(threadName)s %(levelname)s - %(message)s')

    np.random.seed(0)
    tf.random.set_seed(0)

    logging.info(f'Cache location: {memory.location}')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.experiment_id is not None:
        output_dir_full = os.path.join(args.output, args.experiment_id)
        log_dir_full = os.path.join(args.log_dir, args.experiment_id, current_time)
    else:
        output_dir_full = args.output
        log_dir_full = os.path.join(args.log_dir, current_time)
    logging.info(f'Output directory: {output_dir_full}')
    logging.info(f'Log directory: {log_dir_full}')

    subset_names = ('train', 'test')
    summary_writers = {name: tf.summary.create_file_writer(os.path.join(log_dir_full, name)) for name in subset_names}
    tf.summary.experimental.set_step(0)

    with summary_writers['train'].as_default():
        tf.summary.text('args', str(args))

    logging.info('TensorFlow physical devices: %s', tf.config.experimental.list_physical_devices())

    with joblib.parallel_backend('threading', n_jobs=args.jobs), tf.device(args.device):
        solver = Solver(timeout=20)
        graphifier = Graphifier(solver, max_number_of_nodes=args.max_problem_size)
        if args.evaluate_on_training_set:
            eval_dataset_names = ('test', 'train')
        else:
            eval_dataset_names = ('test',)
        problems, eval_datasets = get_data(args.question_dir, args.signature_dir, graphifier, args.cache_file, args.train_size,
                                           args.test_size, BatchGenerator(args.max_test_batch_size),
                                           max_problems=args.max_problems,
                                           max_questions_per_problem=args.max_questions_per_problem,
                                           output_dir=output_dir_full, datasets=eval_dataset_names, device=args.device)
        save_dataset_stats(problems, eval_datasets, output_dir_full)

        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

        k = 12
        w_values = []
        records = []
        if args.evaluate_linear_standard:
            w_values.extend(itertools.chain(np.eye(k), np.eye(k) * -1))
        rng = np.random.RandomState(0)
        w_values.extend(rng.normal(0, 1, k) for _ in range(args.evaluate_linear_random))
        for w in tqdm(w_values, unit='model', desc='Evaluating linear models'):
            record = simple_features.evaluate_weights(w, eval_datasets['test'], eval_datasets['train'], loss_fn)
            records.append(record)
        save_df(utils.dataframe_from_records(records), 'linear_models', output_dir_full)

        optimizers = {
            'sgd': keras.optimizers.SGD,
            'adam': keras.optimizers.Adam,
            'rmsprop': keras.optimizers.RMSprop
        }
        optimizer = optimizers[args.optimizer](learning_rate=args.learning_rate)

        records_evaluation = []
        records_training = []

        try:
            model = SymbolPreferenceGCN('predicate', graphifier.canonical_etypes, graphifier.ntypes)
            if args.plot_model is not None:
                keras.utils.plot_model(model, utils.path_join(output_dir_full, args.plot_model, makedir=True),
                                       show_shapes=True)
            rng = np.random.RandomState(0)
            if args.max_train_batch_size > 0:
                batch_generator_train = BatchGenerator(max_batch_length=args.max_train_batch_size,
                                                       problems_per_batch=args.train_batch_problems)
            else:
                batch_generator_train = BatchGenerator(problems_per_batch=args.train_batch_problems)
            step_i = tf.Variable(0, dtype=tf.int64)
            ckpt = tf.train.Checkpoint(step=step_i, optimizer=optimizer, model=model)
            manager = tf.train.CheckpointManager(ckpt, os.path.join(output_dir_full, 'tf_ckpts', 'auto'),
                                                 max_to_keep=10, step_counter=step_i)
            ckpt.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                logging.info(f'Restored from {manager.latest_checkpoint}.')
            else:
                logging.info('No checkpoint to restore. Training from scratch.')
            best_accuracy = {k: 0 for k in eval_dataset_names}
            with tqdm(unit='step', desc='Training', total=args.steps) as t:
                t.update(int(step_i))
                postfix = {}
                while args.steps is None or step_i < args.steps:
                    tf.summary.experimental.set_step(step_i)
                    if int(step_i) == args.profile_start:
                        tf.profiler.experimental.start(args.log_dir)
                    if int(step_i) == args.profile_stop:
                        try:
                            tf.profiler.experimental.stop()
                        except tf.errors.UnavailableError:
                            logging.warning('Attempting to stop profiling when no profiler is running.', exc_info=True)
                    if int(step_i) % args.evaluation_period == 0:
                        with tf.profiler.experimental.Trace('test', step_num=step_i, _r=1):
                            eval_record = evaluate(model, eval_datasets, loss_fn, summary_writers)
                            postfix.update({'.'.join(k): v for k, v in eval_record.items() if k != 'step'})
                            t.set_postfix(postfix)
                            records_evaluation.append(eval_record)
                            manager.save(checkpoint_number=step_i)
                            for dataset_name, best_value in best_accuracy.items():
                                record_key = (dataset_name, 'accuracy')
                                try:
                                    cur_value = eval_record[record_key]
                                    if cur_value > best_value:
                                        best_accuracy[dataset_name] = cur_value
                                        saved_path = ckpt.write(
                                            os.path.join(output_dir_full, 'tf_ckpts', 'accuracy', dataset_name))
                                        logging.info(
                                            f'New best {dataset_name} accuracy at step {int(step_i)}: {cur_value}. Checkpoint written to {saved_path}.')
                                except KeyError:
                                    pass
                    with tf.profiler.experimental.Trace('train', step_num=step_i, _r=1):
                        with summary_writers['train'].as_default():
                            record = train_step(model, problems['train'], loss_fn, optimizer, rng,
                                                batch_generator_train)
                            postfix['loss'] = record['loss']
                            t.set_postfix(postfix)
                            records_training.append(record)
                    step_i.assign_add(1)
                    t.update()
            i = args.steps
            assert i is not None
            tf.summary.experimental.set_step(i)
            records_evaluation.append(evaluate(model, eval_datasets, loss_fn, summary_writers))
        finally:
            try:
                tf.profiler.experimental.stop()
            except tf.errors.UnavailableError:
                pass
            save_df(
                utils.dataframe_from_records(records_evaluation, index_keys='step', dtypes={'step': pd.UInt32Dtype()}),
                'steps_evaluation', output_dir_full)
            save_df(
                utils.dataframe_from_records(records_training, index_keys='step', dtypes={'step': pd.UInt32Dtype()}),
                'steps_training', output_dir_full)


def save_dataset_stats(problems, eval_datasets, output_dir):
    records = []
    for dataset_name in ('test', 'train'):
        record = {'name': dataset_name}
        cur_problems = problems[dataset_name]
        record['problem', 'count'] = len(cur_problems)
        if len(problems[dataset_name]) >= 1:
            # Cumulative problem size with all questions
            record['problem', 'size', 'all', 'sum'] = sum(map(problem_size, cur_problems))
            # Maximum problem size with all questions
            record['problem', 'size', 'all', 'max'] = max(map(problem_size, cur_problems))
            # Maximum problem size with 1 question
            record['problem', 'size', 1, 'max'] = max(map(lambda p: problem_size(p, n_questions=1), cur_problems))
        dataset = eval_datasets[dataset_name]
        if dataset is not None:
            xs = dataset['xs']
            record['batches'] = len(xs)
            record['ranking_difference', 'len', 'sum'] = sum(len(x['ranking_difference']) for x in xs)
            record['ranking_difference', 'len', 'max'] = max(len(x['ranking_difference']) for x in xs)
            sample_weight = dataset['sample_weight']
            # Number of questions
            record['sample_weight', 'len'] = len(sample_weight)
            record['sample_weight', 'sum'] = np.sum(sample_weight)
        records.append(record)
    logging.info(records)
    dtypes = {
        'name': 'category',
        ('problem', 'count'): pd.UInt32Dtype(),
        ('problem', 'size', 'all', 'sum'): pd.UInt32Dtype(),
        ('problem', 'size', 'all', 'max'): pd.UInt32Dtype(),
        ('problem', 'size', 1, 'max'): pd.UInt32Dtype(),
        'batches': pd.UInt32Dtype(),
        ('ranking_difference', 'len', 'sum'): pd.UInt32Dtype(),
        ('ranking_difference', 'len', 'max'): pd.UInt32Dtype(),
        ('sample_weight', 'len'): pd.UInt32Dtype(),
        ('sample_weight', 'sum'): np.float
    }
    save_df(utils.dataframe_from_records(records, index_keys='name', dtypes=dtypes), 'datasets', output_dir)


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
        question_symbols = x['question_symbols']
        ranking_difference = x['ranking_difference']
        segment_ids = x['segment_ids']
        return self.call_symbol_embeddings(symbol_embeddings, question_symbols, ranking_difference, segment_ids)

    @tf.function(experimental_relax_shapes=True)
    def call_symbol_embeddings(self, symbol_embeddings, question_symbols, ranking_difference, segment_ids):
        # Row: problem -> symbol
        symbol_costs = tf.squeeze(self.cost_model(symbol_embeddings))
        # Row: problem -> question -> symbol
        symbol_costs_tiled = tf.gather(symbol_costs, question_symbols)
        potentials = tf.multiply(symbol_costs_tiled, ranking_difference)
        # Row: problem -> question
        precedence_pair_logit = tf.math.segment_sum(potentials, segment_ids)
        return precedence_pair_logit


# Cannot cache this call because we set device on some objects.
def get_data(question_dir, signature_dir, graphifier, cache_file, train_size, test_size, batch_generator, max_problems,
             max_questions_per_problem, random_state=0, output_dir=None, datasets=None, device=None):
    problems_all = get_problems(question_dir, signature_dir, graphifier, max_problems, max_questions_per_problem,
                                np.random.RandomState(random_state), cache_file, output_dir)
    logging.info(f'Number of problems graphified: {len(problems_all)}')

    # Set device of the graphs
    # We set the device before batching the graphs to save memory.
    if device is not None:
        for v in tqdm(problems_all.values(), desc=f'Moving graphs to device {device}', unit='problem'):
            v['graph'] = v['graph'].to(device)

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


def evaluate(model, datasets, loss_fn, summary_writers=None):
    record = {'step': int(tf.summary.experimental.get_step())}
    for dataset_name, dataset in datasets.items():
        for name, value in test_step(model, dataset, loss_fn).items():
            record[(dataset_name, name)] = value.numpy()
            if summary_writers is not None:
                with summary_writers[dataset_name].as_default():
                    tf.summary.scalar(f'evaluation.{name}', value)
    return record


def train_step(model, problems, loss_fn, optimizer, rng, batch_generator, log_grads=True):
    record = {'step': int(tf.summary.experimental.get_step())}
    x, sample_weight = batch_generator.get_batch_random(problems, rng)
    record['size'] = number_of_nodes(x['batch_graph']) + len(x['ranking_difference'])
    record['problems'] = x['batch_graph'].batch_size
    record['nodes'] = number_of_nodes(x['batch_graph'])
    record['ranking_difference', 'len'] = len(x['ranking_difference'])
    tf.summary.histogram('ranking_difference', x['ranking_difference'])
    record['sample_weight', 'len'] = len(sample_weight)
    record['sample_weight', 'sum'] = np.sum(sample_weight)
    record['sample_weight', 'mean'] = np.mean(sample_weight)
    tf.summary.histogram('sample_weight', sample_weight)
    time_start = time.time()
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        assert len(logits) == len(sample_weight)
        loss_value = loss_fn(np.ones((len(sample_weight), 1), dtype=np.bool), tf.expand_dims(logits, 1),
                             sample_weight=sample_weight)
        # loss_value is average loss over samples (questions).
    record['time', 'grads', 'compute'] = time.time() - time_start
    # Normalize the loss so that it is consistent with the metric BinaryCrossentropy.
    # We still train using the raw loss value to ensure that the sample weights are consistent across training batches.
    record['loss'] = loss_value.numpy() / np.mean(sample_weight)
    grads = tape.gradient(loss_value, model.trainable_weights)
    if log_grads:
        grads_flat = tf.concat([tf.keras.backend.flatten(g) for g in grads if g is not None], 0)
        record['grads', 'mean'] = tf.keras.backend.mean(grads_flat).numpy()
        record['grads', 'norm', 1] = tf.norm(grads_flat, ord=1).numpy()
        record['grads', 'norm', 2] = tf.norm(grads_flat, ord=2).numpy()
        tf.summary.histogram('grads', grads_flat)
    time_start = time.time()
    # https://stackoverflow.com/a/60217922/4054250
    optimizer.apply_gradients(filter(lambda x: x[0] is not None, zip(grads, model.trainable_weights)))
    record['time', 'grads', 'apply'] = time.time() - time_start
    for key, value in record.items():
        if key == 'step':
            continue
        if isinstance(key, tuple):
            key = '.'.join(map(str, key))
        tf.summary.scalar(key, value)
    return record


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
    # We normalize loss by dividing it with the mean sample weight so that the loss is consistent with the metric BinaryCrossentropy.
    res = {'loss': loss_fn(np.ones((len(sample_weight), 1), dtype=np.bool), tf.expand_dims(logits, 1),
                           sample_weight=sample_weight) / np.mean(sample_weight)}
    for name, metric in metrics.items():
        metric.update_state(np.ones((len(sample_weight), 1), dtype=np.bool), tf.expand_dims(logits, 1),
                            sample_weight=sample_weight)
        res[name] = metric.result()
        metric.reset_states()
    return res


def problem_size(d, n_questions=None):
    g = d['graph']
    if n_questions is None:
        n_questions = len(d['questions'])
    assert n_questions <= len(d['questions'])
    return number_of_nodes(g) + d['questions'].shape[1] * n_questions


class BatchGenerator:
    def __init__(self, max_batch_length=None, problems_per_batch=None):
        self.max_batch_length = max_batch_length
        self.problems_per_batch = problems_per_batch

    def __repr__(self):
        return f'{type(self).__name__}({self.max_batch_length})'

    def get_batches(self, problems):
        if len(problems) == 0:
            # This happens namely when the user sets test set size to 0.
            return None
        xs = []
        sample_weight_list = []
        problem_sizes = dict(enumerate(map(problem_size, problems)))
        max_size = max(problem_sizes.values())
        logging.info(f'Maximum problem size: {max_size}')
        if max_size > self.max_batch_length:
            raise RuntimeError(
                f'Failed to distribute problems in batches of size at most {self.max_batch_length}. Problem of size {max_size} encountered.')
        bins = binpacking.to_constant_volume(problem_sizes, self.max_batch_length)
        logging.info(f'Number of batches: {len(bins)}')
        for cur_bin in tqdm(bins, desc='Generating batches', unit='batch'):
            batch_size = sum(cur_bin.values())
            assert batch_size <= self.max_batch_length
            x, sample_weight = self._get_batch(problems[i] for i in cur_bin.keys())
            xs.append(x)
            sample_weight_list.append(sample_weight)
        sample_weight = np.concatenate(sample_weight_list)
        return {'xs': xs, 'sample_weight': sample_weight}

    def get_batch_random(self, problems, rng):
        if self.problems_per_batch is not None and self.problems_per_batch < len(problems):
            problems = rng.choice(problems, self.problems_per_batch, replace=False)
        if self.max_batch_length is None:
            return self._get_batch(problems)
        question_ids = list(itertools.chain.from_iterable(
            ((problem_i, question_i) for question_i in range(len(problem['questions']))) for
            problem_i, problem in enumerate(problems)))
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
            if self.max_batch_length is not None:
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
        x_lists = {'ranking_difference': [], 'question_symbols': [], 'segment_ids': [], 'symbol_embeddings_predicate': [], 'symbol_embeddings_function': []}
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
            x_lists['symbol_embeddings_predicate'].append(d['signatures']['predicate'])
            x_lists['symbol_embeddings_function'].append(d['signatures']['function'])
            sample_weight_list.append(np.full(m, 1 / m, dtype=dtype_tf_float))
            question_i += m
            symbol_i += n
            total_size += problem_size(d, len(questions))
        assert self.max_batch_length is None or total_size <= self.max_batch_length
        x = {k: np.concatenate(v) for k, v in x_lists.items()}
        assert len(x['ranking_difference']) == len(x['question_symbols']) == len(x['segment_ids'])
        assert symbol_i == np.max(x['question_symbols']) + 1
        sample_weight = np.concatenate(sample_weight_list)
        assert len(sample_weight) == question_i
        x['batch_graph'] = dgl.batch(graphs)
        assert x['batch_graph'].number_of_nodes('predicate') == len(x['symbol_embeddings_predicate'])
        assert x['batch_graph'].number_of_nodes('function') == len(x['symbol_embeddings_function'])
        logging.debug(
            'Batch created. Size: %d/%d (actual/maximum). Problems: %d. Nodes: %d. Questions: %d. Cumulative symbol count: %d. Cumulative question length: %d.',
            total_size, self.max_batch_length, len(graphs), number_of_nodes(x['batch_graph']), question_i, symbol_i,
            len(x['ranking_difference']))
        return x, sample_weight


if __name__ == '__main__':
    main()
