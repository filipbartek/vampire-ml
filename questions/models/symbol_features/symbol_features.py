import tensorflow as tf


class SymbolFeatures(tf.keras.layers.Layer):
    def __init__(self, name='problem_to_embedding', **kwargs):
        tf.keras.layers.Layer.__init__(self, name=name, **kwargs)

    def call(self, problems):
        """
        :param problems: Tensor of problem names. Shape: `[None]`. Dtype: `tf.string`.
        :return: Dictionary with two elements:
        - 'embeddings': Ragged tensor of symbol embeddings. Shape `[problems.shape[0], None, n]`
          where `n` is the symbol embedding length.
          Each row is a stack of symbol embeddings for the corresponding problem.
          If the embedding for the given problem fails to be predicted, the corresponding row is empty.
        - 'valid': Tensor of output validity flags. Shape: `[problems.shape[0]]`. Dtype: tf.bool.
        """
        raise NotImplementedError

    def invalid_embedding(self, n):
        return tf.constant(0, dtype=self.dtype, shape=(0, n))
