import tensorflow as tf


class SymbolCostModel(tf.keras.Model):
    def compile(self, metrics, run_eagerly=None):
        super().compile(run_eagerly=run_eagerly)
        self.symbol_cost_metrics = metrics

    def train_step(self, data):
        raise NotImplementedError

    def test_step(self, problems):
        # We assume that only problems are passed, not y nor sample_weight.
        symbol_costs_decorated = self(problems, training=False)
        for m in self.symbol_cost_metrics:
            m.update_state(problems, symbol_costs_decorated['costs'], symbol_costs_decorated['valid'])
        return {m.name: m.result() for m in self.symbol_cost_metrics}
