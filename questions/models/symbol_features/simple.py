import tensorflow as tf

from .symbol_features import SymbolFeatures


class SimpleSymbolFeaturesModel(SymbolFeatures):
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
        fn_output_signature = [tf.RaggedTensorSpec(shape=[None, self.n], dtype=self.dtype, ragged_rank=0), tf.bool]
        res = tf.map_fn(self.predict_one, problems, fn_output_signature=fn_output_signature)
        return {'embeddings': res[0], 'valid': res[1]}

    def predict_one(self, problem):
        # TODO: Cache the problem signatures so that TF need not run the pure Python function.
        return tf.py_function(self._predict_one, [problem], (self.dtype, tf.bool))

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
            return df_filtered.to_numpy(dtype=self.numpy_dtype), True
        except RuntimeError:
            # If the solver fails to determine the problem signature, one row with all nans is returned.
            return self.invalid_embedding(self.n), False

    @property
    def numpy_dtype(self):
        return tf.dtypes.as_dtype(self.dtype).as_numpy_dtype
