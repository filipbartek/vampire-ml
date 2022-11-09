import tensorflow as tf


class Dense(tf.keras.layers.Dense):
    def __init__(self, *args, output_bias=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_bias = output_bias

    # Dense layer that supports ragged tensor input.
    def call(self, inputs):
        if isinstance(inputs, tf.RaggedTensor):
            # Note: The last dimension is assumed to be non-ragged.
            res = tf.RaggedTensor.from_nested_row_splits(super().call(inputs.flat_values), inputs.nested_row_splits)
        else:
            res = super().call(inputs)
        if self.output_bias is not None:
            res += self.output_bias
        return res
