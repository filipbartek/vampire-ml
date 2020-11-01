import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.utils.tf_utils as tf_utils
from tqdm import tqdm

from proving import config
from proving import file_path_list
from proving import load_questions
from proving.solver import Solver


class QuestionLogitModel(tf.keras.Model):
    def __init__(self, symbol_cost_model, update_symbol_cost_metrics=False):
        super().__init__()
        self.symbol_cost_model = symbol_cost_model
        self.update_symbol_cost_metrics = update_symbol_cost_metrics

    @staticmethod
    def raggify_questions(questions):
        if isinstance(questions, dict):
            flat_values = questions['flat_values']
            nested_row_splits = questions['nested_row_splits']
            if len(flat_values.shape) == 2:
                flat_values = tf.squeeze(flat_values, axis=1)
                nested_row_splits = tuple(tf.squeeze(e, axis=1) for e in nested_row_splits)
            questions = tf.RaggedTensor.from_nested_row_splits(flat_values, nested_row_splits)
        return questions

    def reset_metrics(self):
        super().reset_metrics()
        self.symbol_cost_model.reset_metrics()

    def test_step(self, x):
        # We assume that only x is passed, not y nor sample_weight.
        questions = self.raggify_questions(x['questions'])
        y = tf.zeros((questions.row_splits[-1], 1))
        sample_weight = tf.concat([tf.fill(l, 1 / l) for l in questions.row_lengths()], 0)
        x['questions'] = questions

        y_pred = tf.reshape(self(x, training=False).flat_values, (-1, 1))

        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        # TODO: Scale binary_crossentropy to match loss.
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        res = {m.name: m.result() for m in self.metrics}

        if self.update_symbol_cost_metrics and self.symbol_cost_model.compiled_metrics is not None:
            symbol_cost_results = self.symbol_cost_model.test_step(x['problems'])
            res.update({k: v for k, v in symbol_cost_results.items() if k != 'loss'})

        return res

    def train_step(self, x):
        questions = self.raggify_questions(x['questions'])
        y = tf.zeros((questions.row_splits[-1], 1))
        sample_weight = tf.concat([tf.fill(l, 1 / l) for l in questions.row_lengths()], 0)
        x['questions'] = questions

        with tf.GradientTape() as tape:
            y_pred = tf.reshape(self(x, training=False).flat_values, (-1, 1))
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        tf.python.keras.engine.training._minimize(self.distribute_strategy, tape, self.optimizer, loss,
                                                  self.trainable_variables)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        res = {m.name: m.result() for m in self.metrics}

        if self.update_symbol_cost_metrics and self.symbol_cost_model.compiled_metrics is not None:
            symbol_cost_results = self.symbol_cost_model.test_step(x['problems'])
            res.update({k: v for k, v in symbol_cost_results.items() if k != 'loss'})

        return res

    def call(self, x, training=False):
        problems = x['problems']
        questions = self.raggify_questions(x['questions'])
        symbol_costs = self.symbol_cost_model(problems, training=training)
        logits = self.costs_to_logits(symbol_costs, questions)
        return logits

    @staticmethod
    @tf.function
    def costs_to_logits(symbol_costs, questions):
        # assert symbol_costs.nrows() == questions.nrows()
        starts = tf.repeat(symbol_costs.row_splits[:-1], questions.row_lengths())
        limits = tf.repeat(symbol_costs.row_splits[1:], questions.row_lengths())
        indices = tf.ragged.range(starts, limits)
        sc_tiled = tf.gather(symbol_costs.flat_values, indices)
        sc_like_questions = tf.RaggedTensor.from_row_splits(sc_tiled, questions.row_splits)
        # assert all(tf.reduce_all(sc_split == q_split) for sc_split, q_split in zip(sc_like_questions.nested_row_splits, questions.nested_row_splits))
        potentials = tf.ragged.map_flat_values(tf.multiply, questions, sc_like_questions)
        logits = tf.reduce_sum(potentials, axis=2)
        # assert (logits.shape + (None,)).as_list() == questions.shape.as_list()
        # assert tf.reduce_all(logits.row_splits == questions.row_splits)
        return logits


class SymbolCostModel(tf.keras.Model):
    def __init__(self, problem_to_embedding, embedding_to_cost=None):
        super().__init__()
        self.problem_to_embedding = problem_to_embedding
        if embedding_to_cost is None:
            embedding_to_cost = tf.keras.layers.Dense(1)
        # TimeDistributed handles the ragged dimension 1 of symbol embeddings.
        # https://github.com/tensorflow/tensorflow/issues/39072#issuecomment-631759113
        self.embedding_to_cost = tf.keras.layers.TimeDistributed(embedding_to_cost)

    def test_step(self, data):
        if not isinstance(data, tuple):
            data = (data, data)
        return super().test_step(data)

    def call(self, problems, training=False):
        # `Model.evaluate` pads `problems` with a length 1 axis.
        if len(problems.shape) == 2:
            problems = tf.squeeze(problems, axis=1)
        embeddings = self.problem_to_embedding(problems, training=training)
        costs = self.embedding_to_cost(embeddings, training=training)
        costs = tf.squeeze(costs, axis=2)
        assert costs.shape[0] == len(problems)
        return costs


class SimpleSymbolFeaturesModel(tf.keras.layers.Layer):
    def __init__(self, solver, symbol_type, columns=None, dtype=None):
        super().__init__(trainable=False, dtype=dtype)
        self.solver = solver
        self.symbol_type = symbol_type
        self.columns = columns

    @property
    def n(self):
        if self.columns is None:
            return 12
        else:
            return len(self.columns)

    def call(self, problems):
        problem_embeddings = tf.TensorArray(self.dtype, size=len(problems), dynamic_size=False,
                                            colocate_with_first_write_call=False, infer_shape=False)
        row_lengths_array = tf.TensorArray(tf.int32, size=len(problems), dynamic_size=False,
                                           colocate_with_first_write_call=False)
        # TODO: Parallelize.
        for i in tf.range(len(problems)):
            problem = problems[i]
            one = self.predict_one(problem)
            # assert one.shape[1] == self.n
            problem_embeddings = problem_embeddings.write(i, one)
            row_lengths_array = row_lengths_array.write(i, len(one))
        # assert all(e.shape[1] == problem_embeddings[0].shape[1] for e in problem_embeddings)
        values = problem_embeddings.concat()
        row_lengths = row_lengths_array.stack()
        res = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        return res

    def predict_one(self, problem):
        return tf.py_function(self._predict_one, [problem], self.dtype)

    def _predict_one(self, problem):
        # https://stackoverflow.com/a/56122892/4054250
        problem = bytes.decode(problem.numpy())
        try:
            df = self.solver.symbols_of_type(problem, self.symbol_type)
            if self.columns is not None:
                df_filtered = df[self.columns]
            else:
                df_filtered = df.drop('name', axis='columns')
            assert df_filtered.shape[1] == self.n
            return df_filtered.to_numpy(dtype=self.numpy_dtype)
        except RuntimeError:
            # If the solver fails to determine the problem signature, one row with all nans is returned.
            return tf.constant(np.nan, dtype=self.dtype, shape=(1, self.n))

    @property
    def numpy_dtype(self):
        return tf.dtypes.as_dtype(self.dtype).as_numpy_dtype


class SolverSuccessRate(tf.keras.metrics.Mean):
    def __init__(self, solver, symbol_type, name='solver_success_rate', **kwargs):
        super().__init__(name=name, **kwargs)
        self.solver = solver
        self.symbol_type = symbol_type

    def update_state(self, problems, symbol_costs, sample_weight=None):
        if len(problems.shape) == 2:
            problems = tf.squeeze(problems, axis=1)
        values = []
        for problem, symbol_cost in zip(problems, symbol_costs):
            if tf.reduce_any(tf.math.is_nan(symbol_cost)):
                values.append(False)
            else:
                precedence = tf.argsort(symbol_cost)
                succ = self.solve_one(problem, precedence)
                values.append(succ)
        super().update_state(values, sample_weight=sample_weight)

    def solve_one(self, problem, precedence):
        return tf.py_function(self._solve_one, [problem, precedence], tf.bool)

    def _solve_one(self, problem, precedence):
        problem = bytes.decode(problem.numpy())
        precedence = precedence.numpy()
        solver_res = self.solver.solve(problem, precedences={self.symbol_type: precedence})
        # TODO: Collect more detailed stats, namely timeouts (`returncode is None`).
        return solver_res.returncode == 0


def problem_path_to_name(path):
    base_name = tf.strings.split(path, sep='/')[-1]
    problem_name = tf.strings.regex_replace(base_name,
                                            r'^(?P<problem_name>[A-Z]{3}[0-9]{3}[-+^=_][1-9][0-9]*(?:\.[0-9]{3})*)\.p$',
                                            r'\1', False)
    return problem_name


def get_problems_dataset(patterns, shuffle=True, seed=0):
    base_path = config.problems_path()
    patterns_normalized = [file_path_list.normalize_path(pattern, base_path=base_path) for pattern in patterns]
    dataset_files = tf.data.Dataset.list_files(patterns_normalized, shuffle=shuffle, seed=seed)
    return dataset_files.map(problem_path_to_name)


problems_diverse = [
    'PUZ001+1',  # PUZ001+
    'PUZ001-1',  # PUZ001-
    'MSC014+1',  # Solvable quickly
    'LAT337+1',  # 50 % solvable
    'NLP026+1',  # Highest variation
    'LDA005-1',  # Unsolvable (?)
    'SWV540-1.007',  # Broke a job
    'HWV134-1',  # size: 276455099 (max), clauses: 2332428 (max), atoms: 6570884 (max), variables: 4034129 (max)
    'HWV134+1',  # size: 88075841 (max out of '+')
    'HWV092-1',  # size: 85651816 (max out of rating 0), rating: 0
    'BIO001+1',  # atoms_equality: 121152 (max)
    'SYO599-1',  # clause_size_max: 1581 (max)
    'HWV132-1',  # predicates: 480215 (max)
    'HWV061-1',  # predicates_propositional: 10166 (max)
    'SYO601-1',  # predicates_arity_max: 350 (max)
    'LCL649+1.020',  # functors: 0 (min), predicates: 421 (max out of functors 0)
    'SYN915-1',  # size: 1110 (min), functors: 0 (min)
]


def get_questions_datasets(question_dir, problems, cache_dir=None):
    res = {}
    for k, p in problems.items():
        questions = load_questions.get_dataset(question_dir, p)
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, k)
            logging.info('Caching into: %s', cache_path)
            # Parameters: problems, question set (path), dataset (train or validation), batch size
            questions = questions.cache(cache_path)
        res[k] = questions
    return res


def get_questions_datasets_batched(question_dir, problems, batch_size, cache_dir=None):
    logging.info('Batch size: %d', batch_size)
    questions = get_questions_datasets(question_dir, problems, cache_dir)
    res = {}
    for k, q in questions.items():
        batches = load_questions.questions_batched_dataset(q, batch_size)
        res[k] = batches
    return res


def preload_dataset(dataset):
    n_batches = 0
    n_elements = 0
    for batch in tqdm(dataset, unit='batch', desc=f'Preloading batches'):
        n_batches += 1
        n_elements += len(batch['problems'])
    logging.info(f'Number of batches: {n_batches}')
    logging.info(f'Number of problems with questions: {n_elements}')


def get_problems_datasets(patterns, validation_split, max_problems=None):
    problems_all = get_problems_dataset(patterns)
    logging.info('Number of problems available: %d', problems_all.cardinality())
    logging.debug('Leading 10 problems: %s', [bytes.decode(p.numpy()) for p in problems_all.take(10)])
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


class SymbolCostEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, problems=None, problems_validation=None, start=0, step=1):
        super().__init__()
        self.problems = problems
        self.problems_validation = problems_validation
        self.start = start
        self.step = step

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start and (epoch - self.start) % self.step == 0:
            assert logs is not None
            symbol_cost_model = self.model.symbol_cost_model
            if self.problems is not None:
                print('Evaluating symbol cost model on training problems...')
                train_res = symbol_cost_model.evaluate(self.problems, return_dict=True)
                logs.update({k: v for k, v in train_res.items() if k != 'loss'})
            if self.problems_validation is not None:
                print('Evaluating symbol cost model on validation problems...')
                validation_res = symbol_cost_model.evaluate(self.problems_validation, return_dict=True)
                logs.update({f'val_{k}': v for k, v in validation_res.items() if k != 'loss'})
            print(f'Metrics after epoch {epoch}: {logs}')


def main():
    logging.basicConfig(level=logging.INFO)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(True)

    patterns = ['**/*-*.p', '**/*+*.p']
    validation_split = 0.5
    max_problems = 100
    question_dir = '/home/filip/projects/vampire-ml/questions/pred_10k'
    batch_size = 10
    symbol_type = 'predicate'

    solver = Solver(timeout=20)

    problems = get_problems_datasets(patterns, validation_split, max_problems)
    questions = get_questions_datasets_batched(question_dir, problems, batch_size, os.path.join('cache', 'questions'))
    for k, q in questions.items():
        logging.info(f'Preloading {k} batches...')
        preload_dataset(q)

    model_simple = SimpleSymbolFeaturesModel(solver, symbol_type)
    model_symbol_cost = SymbolCostModel(model_simple)
    model_symbol_cost.compile(metrics=[SolverSuccessRate(solver, symbol_type)], run_eagerly=True)
    model_logit = QuestionLogitModel(model_symbol_cost)
    model_logit.compile(optimizer='adam',
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0),
                                 tf.keras.metrics.BinaryCrossentropy(from_logits=True)],
                        run_eagerly=True)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch='10,20'),
        SymbolCostEvaluationCallback(problems=problems['train'].batch(1),
                                     problems_validation=problems['validation'].batch(1),
                                     step=10)
    ]
    logging.info('Training...')
    fit_res = model_logit.fit(questions['train'], validation_data=questions['validation'], epochs=100,
                              callbacks=callbacks)
    print(fit_res)


if __name__ == '__main__':
    main()
