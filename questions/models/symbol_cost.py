import tensorflow as tf


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
