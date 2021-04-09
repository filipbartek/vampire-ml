import tensorflow as tf

from .symbol_cost import SymbolCostModel


class Baseline(SymbolCostModel):
    def __init__(self, name='baseline'):
        super().__init__(name=name)

    def call(self, problems):
        # `Model.test_step` pads `problems` with a length 1 axis.
        if len(problems.shape) == 2:
            problems = tf.squeeze(problems, axis=1)
        n_problems = tf.shape(problems)[0]
        costs = tf.RaggedTensor.from_row_lengths(tf.zeros((0,), dtype=self.dtype),
                                                 tf.zeros((n_problems,), dtype=tf.int64))
        return {'costs': costs, 'valid': tf.ones((n_problems,), dtype=tf.bool)}
