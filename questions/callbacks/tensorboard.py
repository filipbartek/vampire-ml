import tensorflow as tf


class TensorBoard(tf.keras.callbacks.TensorBoard):
    @property
    def train_writer(self):
        return self._train_writer

    @property
    def val_writer(self):
        return self._val_writer

    def writer(self, name):
        return getattr(self, f'{name}_writer')

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        with self.train_writer.as_default():
            tf.summary.scalar('learning_rate', self.model.optimizer.learning_rate, step=epoch)
