import tensorflow as tf


class Dense(tf.keras.layers.Dense):
    # Dense layer that supports ragged tensor input.
    def call(self, inputs):
        if isinstance(inputs, tf.RaggedTensor):
            # Note: The last dimension is assumed to be non-ragged.
            return tf.RaggedTensor.from_nested_row_splits(super().call(inputs.flat_values), inputs.nested_row_splits)
        return super().call(inputs)
