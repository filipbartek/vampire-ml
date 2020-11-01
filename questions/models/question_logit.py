import tensorflow as tf
import tensorflow.python.keras.utils.tf_utils as tf_utils


class QuestionLogitModel(tf.keras.Model):
    def __init__(self, symbol_cost_model, update_symbol_cost_metrics=False):
        super().__init__()
        self.symbol_cost_model = symbol_cost_model
        self.update_symbol_cost_metrics = update_symbol_cost_metrics

    @staticmethod
    def raggify_questions(questions):
        if isinstance(questions, dict):
            flat_values = questions['flat_values']
            nested_row_splits = questions['nested_row_splits']
            if len(flat_values.shape) == 2:
                flat_values = tf.squeeze(flat_values, axis=1)
                nested_row_splits = tuple(tf.squeeze(e, axis=1) for e in nested_row_splits)
            questions = tf.RaggedTensor.from_nested_row_splits(flat_values, nested_row_splits)
        return questions

    def reset_metrics(self):
        super().reset_metrics()
        self.symbol_cost_model.reset_metrics()

    def test_step(self, x):
        # We assume that only x is passed, not y nor sample_weight.
        questions = self.raggify_questions(x['questions'])
        y = tf.zeros((questions.row_splits[-1], 1))
        sample_weight = tf.concat([tf.fill(l, 1 / l) for l in questions.row_lengths()], 0)
        x['questions'] = questions

        y_pred = tf.reshape(self(x, training=False).flat_values, (-1, 1))

        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        # TODO: Scale binary_crossentropy to match loss.
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        res = {m.name: m.result() for m in self.metrics}

        if self.update_symbol_cost_metrics and self.symbol_cost_model.compiled_metrics is not None:
            symbol_cost_results = self.symbol_cost_model.test_step(x['problems'])
            res.update({k: v for k, v in symbol_cost_results.items() if k != 'loss'})

        return res

    def train_step(self, x):
        questions = self.raggify_questions(x['questions'])
        y = tf.zeros((questions.row_splits[-1], 1))
        sample_weight = tf.concat([tf.fill(l, 1 / l) for l in questions.row_lengths()], 0)
        x['questions'] = questions

        with tf.GradientTape() as tape:
            y_pred = tf.reshape(self(x, training=False).flat_values, (-1, 1))
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        tf.python.keras.engine.training._minimize(self.distribute_strategy, tape, self.optimizer, loss,
                                                  self.trainable_variables)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        res = {m.name: m.result() for m in self.metrics}

        if self.update_symbol_cost_metrics and self.symbol_cost_model.compiled_metrics is not None:
            symbol_cost_results = self.symbol_cost_model.test_step(x['problems'])
            res.update({k: v for k, v in symbol_cost_results.items() if k != 'loss'})

        return res

    def call(self, x, training=False):
        problems = x['problems']
        questions = self.raggify_questions(x['questions'])
        symbol_costs = self.symbol_cost_model(problems, training=training)
        logits = self.costs_to_logits(symbol_costs, questions)
        return logits

    @staticmethod
    @tf.function
    def costs_to_logits(symbol_costs, questions):
        # assert symbol_costs.nrows() == questions.nrows()
        starts = tf.repeat(symbol_costs.row_splits[:-1], questions.row_lengths())
        limits = tf.repeat(symbol_costs.row_splits[1:], questions.row_lengths())
        indices = tf.ragged.range(starts, limits)
        sc_tiled = tf.gather(symbol_costs.flat_values, indices)
        sc_like_questions = tf.RaggedTensor.from_row_splits(sc_tiled, questions.row_splits)
        # assert all(tf.reduce_all(sc_split == q_split) for sc_split, q_split in zip(sc_like_questions.nested_row_splits, questions.nested_row_splits))
        potentials = tf.ragged.map_flat_values(tf.multiply, questions, sc_like_questions)
        logits = tf.reduce_sum(potentials, axis=2)
        # assert (logits.shape + (None,)).as_list() == questions.shape.as_list()
        # assert tf.reduce_all(logits.row_splits == questions.row_splits)
        return logits


class SymbolCostEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, problems=None, problems_validation=None, start=0, step=1):
        super().__init__()
        self.problems = problems
        self.problems_validation = problems_validation
        self.start = start
        self.step = step

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start and (epoch - self.start) % self.step == 0:
            assert logs is not None
            symbol_cost_model = self.model.symbol_cost_model
            if self.problems is not None:
                print('Evaluating symbol cost model on training problems...')
                train_res = symbol_cost_model.evaluate(self.problems, return_dict=True)
                logs.update({k: v for k, v in train_res.items() if k != 'loss'})
            if self.problems_validation is not None:
                print('Evaluating symbol cost model on validation problems...')
                validation_res = symbol_cost_model.evaluate(self.problems_validation, return_dict=True)
                logs.update({f'val_{k}': v for k, v in validation_res.items() if k != 'loss'})
            print(f'Metrics after epoch {epoch}: {logs}')
