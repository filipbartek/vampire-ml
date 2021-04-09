import numpy as np
import tensorflow as tf

from questions.utils import py_str
from .symbol_cost import SymbolCostModel


class Random(SymbolCostModel):
    def __init__(self, clausifier, symbol_type, rng, name='random'):
        super().__init__(name=name)
        self.clausifier = clausifier
        self.symbol_type = symbol_type
        self.rng = rng

    def call(self, problems, **kwargs):
        # For each problem, outputs a random permutation of the same length as the problem signature size.

        # `Model.test_step` pads `problems` with a length 1 axis.
        if len(problems.shape) == 2:
            problems = tf.squeeze(problems, axis=1)
        fn_output_signature = [tf.RaggedTensorSpec(shape=[None], dtype=self.dtype, ragged_rank=0),
                               tf.TensorSpec(tf.TensorShape([]), dtype=tf.bool)]
        res = tf.map_fn(self.predict_one, problems, fn_output_signature=fn_output_signature)
        return {'costs': res[0], 'valid': res[1]}

    def predict_one(self, problem):
        return tf.py_function(self._predict_one, [problem], (self.dtype, tf.bool))

    def _predict_one(self, problem):
        problem = py_str(problem)
        try:
            n = len(self.clausifier.symbols_of_type(problem, self.symbol_type))
            cost = self.rng.permutation(n) / (n - 1) - 0.5
            assert np.isclose(0, cost.mean(), rtol=0)
            # The normalization ensures a n-uniform variance of the precedence cost. (?)
            assert np.isclose(0, np.outer(cost, cost).sum(), rtol=0)
            return cost, True
        except RuntimeError:
            # If the solver fails to determine the problem signature, invalid costs are returned.
            return self.invalid_costs(), False

    def invalid_costs(self):
        return tf.constant(0, dtype=self.dtype, shape=(0,))
