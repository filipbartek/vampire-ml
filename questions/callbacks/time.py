import tensorflow as tf


class Timer(tf.keras.metrics.Sum):
    def __init__(self):
        super().__init__()
        self.time_begin = None

    def pop_result(self):
        res = self.result()
        self.reset_states()
        return res

    def begin(self):
        self.time_begin = tf.timestamp()

    def end(self):
        self.update_state(tf.timestamp() - self.time_begin)


class Time(tf.keras.callbacks.Callback):
    def __init__(self, problems, tensorboard):
        super().__init__()
        self.problems = problems
        self.tensorboard = tensorboard
        self.timer_epoch = Timer()
        self.timer_epoch_complete = Timer()
        self.timer_test = Timer()
        self.timer_test_batch = Timer()
        self.timer_train_batch = Timer()

    def on_train_begin(self, logs=None):
        self.timer_epoch_complete.begin()
        super().on_train_begin(logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.timer_epoch.begin()
        super().on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        self.timer_epoch.end()
        with self.tensorboard.train_writer.as_default():
            time_epoch = self.timer_epoch.pop_result()
            tf.summary.scalar('time_epoch', time_epoch, step=epoch)
            self.timer_epoch_complete.end()
            tf.summary.scalar('time_epoch_complete', self.timer_epoch_complete.pop_result(), step=epoch)
            self.timer_epoch_complete.begin()
            tf.summary.scalar('time_batch_sum', self.timer_train_batch.pop_result(), step=epoch)
            tf.summary.scalar('time', time_epoch - self.timer_test.result(), step=epoch)
            self.log_symbol_cost_histogram(self.problems['train'], epoch)
        with self.tensorboard.val_writer.as_default():
            tf.summary.scalar('time', self.timer_test.pop_result(), step=epoch)
            tf.summary.scalar('time_batch_sum', self.timer_test_batch.pop_result(), step=epoch)
            self.log_symbol_cost_histogram(self.problems['validation'], epoch)

    def log_symbol_cost_histogram(self, problems, epoch):
        symbol_costs_decorated = self.model.symbol_cost_model(problems)
        symbol_costs_valid = tf.ragged.boolean_mask(symbol_costs_decorated['costs'],
                                                    symbol_costs_decorated['valid'])
        tf.summary.histogram('sample_symbol_costs', symbol_costs_valid.flat_values, step=epoch)

    def on_test_begin(self, logs=None):
        self.timer_test.begin()
        super().on_test_begin(logs=logs)

    def on_test_end(self, logs=None):
        super().on_test_end(logs=logs)
        self.timer_test.end()

    def on_test_batch_begin(self, batch, logs=None):
        self.timer_test_batch.begin()
        super().on_test_batch_begin(batch, logs=logs)

    def on_test_batch_end(self, batch, logs=None):
        super().on_test_batch_end(batch, logs=logs)
        self.timer_test_batch.end()

    def on_train_batch_begin(self, batch, logs=None):
        self.timer_train_batch.begin()
        super().on_train_batch_begin(batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs=logs)
        self.timer_train_batch.end()