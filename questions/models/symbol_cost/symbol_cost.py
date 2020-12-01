import tensorflow as tf


class SymbolCostModel(tf.keras.Model):
    def __init__(self, name='symbol_cost'):
        super().__init__(name=name)

    def compile(self, solver_metric=None, metrics=None, **kwargs):
        super().compile(**kwargs)
        self.solver_metric = solver_metric
        if metrics is None:
            metrics = []
        if solver_metric is not None and solver_metric not in metrics:
            metrics = [solver_metric] + metrics
        self.symbol_cost_metrics = metrics

    def test_step(self, problems):
        # We assume that only problems are passed, not y nor sample_weight.
        logs = super().test_step(problems)
        symbol_costs_decorated = self(problems, training=False)
        for metric in self.symbol_cost_metrics:
            metric.update_state(problems, symbol_costs_decorated['costs'], symbol_costs_decorated['valid'])
            logs[metric.name] = metric.result()
        return logs


class Baseline(SymbolCostModel):
    def call(self, problems):
        # `Model.test_step` pads `problems` with a length 1 axis.
        if len(problems.shape) == 2:
            problems = tf.squeeze(problems, axis=1)
        n_problems = tf.shape(problems)[0]
        costs = tf.RaggedTensor.from_row_lengths(tf.zeros((0,), dtype=self.dtype),
                                                 tf.zeros((n_problems,), dtype=tf.int64))
        return {'costs': costs, 'valid': tf.ones((n_problems,), dtype=tf.bool)}
