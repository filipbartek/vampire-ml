import numpy as np
import tensorflow as tf


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
