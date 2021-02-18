import tensorflow as tf

from .symbol_features import SymbolFeatures
from proving.utils import py_str


class Simple(SymbolFeatures):
    default_columns = [
        'arity',
        'usageCnt',
        'unitUsageCnt',
        'inGoal',
        'inUnit',
        'skolem',
        'inductionSkolem',
        'interpreted',
        'introduced',
        'stringConstant',
        'numericConstant',
        'interpretedNumber'
    ]

    def __init__(self, clausifier, symbol_type, columns=None, dtype=None):
        super().__init__(trainable=False, dtype=dtype)
        self.clausifier = clausifier
        self.symbol_type = symbol_type
        if columns is None:
            columns = self.default_columns
        self.columns = columns

    @property
    def n(self):
        if self.columns is None:
            return 12
        else:
            return len(self.columns)

    def call(self, problems, cache=True):
        # The output shape of 'valid' is fully specified so that it can be later used as a boolean mask with `tf.ragged.boolean_mask`.
        fn_output_signature = [tf.RaggedTensorSpec(shape=[None, self.n], dtype=self.dtype, ragged_rank=0),
                               tf.TensorSpec(tf.TensorShape([]), dtype=tf.bool)]
        res = tf.map_fn(self.predict_one, problems, fn_output_signature=fn_output_signature)
        return {'embeddings': res[0], 'valid': res[1]}

    def predict_one(self, problem):
        # TODO: Cache the problem signatures so that TF need not run the pure Python function.
        return tf.py_function(self._predict_one, [problem], (self.dtype, tf.bool))

    def _predict_one(self, problem):
        # https://stackoverflow.com/a/56122892/4054250
        problem = py_str(problem)
        try:
            df = self.clausifier.symbols_of_type(problem, self.symbol_type)
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
