import tensorflow as tf


class SolverSuccessRate(tf.keras.metrics.Mean):
    def __init__(self, solver, symbol_type, name='solver_success_rate', **kwargs):
        super().__init__(name=name, **kwargs)
        self.solver = solver
        self.symbol_type = symbol_type

    def update_state(self, problems, symbol_costs, valid, sample_weight=None):
        if len(problems.shape) == 2:
            problems = tf.squeeze(problems, axis=1)
        values = tf.map_fn(self.evaluate_one, (problems, symbol_costs, valid), fn_output_signature=tf.bool)
        super().update_state(values, sample_weight=sample_weight)

    def evaluate_one(self, x):
        problem, symbol_cost, valid = x
        if valid:
            precedence = tf.argsort(symbol_cost)
            succ = self.solve_one(problem, precedence)
            return succ
        else:
            return False

    def solve_one(self, problem, precedence):
        return tf.py_function(self._solve_one, [problem, precedence], tf.bool)

    def _solve_one(self, problem, precedence):
        problem = bytes.decode(problem.numpy())
        precedence = precedence.numpy()
        solver_res = self.solver.solve(problem, precedences={self.symbol_type: precedence})
        # TODO: Collect more detailed stats, namely timeouts (`returncode is None`).
        return solver_res.returncode == 0
