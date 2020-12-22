import tensorflow as tf


class TensorBoard(tf.keras.callbacks.TensorBoard):
    @property
    def train_writer(self):
        return self._train_writer

    @property
    def val_writer(self):
        return self._val_writer

    @property
    def writers(self):
        return {'train': self.train_writer, 'val': self.val_writer}
