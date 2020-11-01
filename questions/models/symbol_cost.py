import tensorflow as tf


class SymbolCostModel(tf.keras.Model):
    def __init__(self, problem_to_embedding, embedding_to_cost=None):
        super().__init__()
        self.problem_to_embedding = problem_to_embedding
        if embedding_to_cost is None:
            embedding_to_cost = tf.keras.layers.Dense(1)
        self.embedding_to_cost = embedding_to_cost

    def test_step(self, data):
        if not isinstance(data, tuple):
            data = (data, data)
        return super().test_step(data)

    def call(self, problems, training=False):
        # `Model.evaluate` pads `problems` with a length 1 axis.
        if len(problems.shape) == 2:
            problems = tf.squeeze(problems, axis=1)
        embeddings = self.problem_to_embedding(problems, training=training)
        # `TimeDistributed` can be used in eager mode to support ragged tensor input.
        # In non-eager mode it is necessary to flatten the ragged tensor `embedding`.
        costs_flat_values = self.embedding_to_cost(embeddings.flat_values, training=training)
        costs_flat_values = tf.squeeze(costs_flat_values, axis=1)
        costs = tf.RaggedTensor.from_nested_row_splits(costs_flat_values, embeddings.nested_row_splits)
        return costs


class SolverSuccessRate(tf.keras.metrics.Mean):
    def __init__(self, solver, symbol_type, name='solver_success_rate', **kwargs):
        super().__init__(name=name, **kwargs)
        self.solver = solver
        self.symbol_type = symbol_type

    def update_state(self, problems, symbol_costs, sample_weight=None):
        if len(problems.shape) == 2:
            problems = tf.squeeze(problems, axis=1)
        values = tf.map_fn(self.evaluate_one, (problems, symbol_costs), fn_output_signature=tf.bool)
        super().update_state(values, sample_weight=sample_weight)

    def evaluate_one(self, x):
        problem, symbol_cost = x
        if tf.reduce_any(tf.math.is_nan(symbol_cost)):
            return False
        else:
            precedence = tf.argsort(symbol_cost)
            succ = self.solve_one(problem, precedence)
            return succ

    def solve_one(self, problem, precedence):
        return tf.py_function(self._solve_one, [problem, precedence], tf.bool)

    def _solve_one(self, problem, precedence):
        problem = bytes.decode(problem.numpy())
        precedence = precedence.numpy()
        solver_res = self.solver.solve(problem, precedences={self.symbol_type: precedence})
        # TODO: Collect more detailed stats, namely timeouts (`returncode is None`).
        return solver_res.returncode == 0
