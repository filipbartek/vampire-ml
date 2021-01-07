import pandas as pd
import tensorflow as tf


class Weights(tf.keras.callbacks.Callback):
    def __init__(self, tensorboard):
        super().__init__()
        self.tensorboard = tensorboard

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        with self.tensorboard.train_writer.as_default():
            symbol_cost_model = self.model.symbol_cost_model
            embedding_to_cost = symbol_cost_model.embedding_to_cost
            kernel = tf.squeeze(embedding_to_cost.kernel, axis=1).numpy()
            columns = symbol_cost_model.problem_to_embedding.columns
            series = pd.Series(dict(zip(columns, kernel)), name='coefficient')
            series.index.name = 'feature'
            tf.summary.text('symbol_cost/kernel', series.to_markdown(), step=epoch)
            for feature, coefficient in zip(columns, kernel):
                # 'symbol_cost_kernel' is a dedicated TensorBoard category because we are interested in comparing the kernel components.
                tf.summary.scalar(f'symbol_cost_kernel/{feature}', coefficient, step=epoch)
            tf.summary.scalar(f'symbol_cost/bias', tf.squeeze(embedding_to_cost.bias, axis=0), step=epoch)
