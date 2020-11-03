import tensorflow as tf


class SymbolFeatures(tf.keras.layers.Layer):
    def call(self, problems):
        """
        :param problems: Tensor of problem names. Shape: `[None]`. Dtype: `tf.string`.
        :return: Ragged tensor of symbol embeddings. Shape `[problems.shape[0], None, n]`
        where `n` is the symbol embedding length.
        Each row is a stack of symbol embeddings for the corresponding problem.
        """
        raise NotImplementedError
