import numpy as np
import sklearn
import tensorflow as tf

from .symbol_cost import SymbolCostModel


class Direct(SymbolCostModel):
    def __init__(self, questions):
        super().__init__()
        self.costs = {}
        for p, q in questions.items():
            self.costs[p] = self.optimize_cost(q)

    def call(self, problems):
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
        problem = bytes.decode(problem.numpy())
        try:
            return self.costs[problem], True
        except KeyError:
            return self.invalid_costs(), False

    def invalid_costs(self):
        return tf.constant(0, dtype=self.dtype, shape=(0,))

    def optimize_cost(self, questions, normalize=True):
        if normalize:
            questions = sklearn.preprocessing.normalize(questions)
        cost = np.mean(questions, axis=0)
        assert np.mean(np.dot(questions, cost) > 0) >= 0.5
        cost = tf.constant(cost, dtype=self.dtype)
        return cost
