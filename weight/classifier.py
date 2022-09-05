import tensorflow as tf


class Classifier(tf.keras.Model):
    def __init__(self, symbol_weight_model, name='classifier'):
        super().__init__(name=name)
        self.symbol_weight_model = symbol_weight_model

    def compile(self, weighted_metrics=None, **kwargs):
        """
        The loss is fixed to BinaryCrossentropy.
        The following weighted metrics are always used:
        - BinaryCrossentropy
        - BinaryAccuracy
        The reported loss metric is inaccurate because it averages over questions.
        Batches with more questions contribute more, while the contribution should only depend on the number of problems.
        The metric BinaryCrossentropy reports correct values.
        """
        if weighted_metrics is None:
            weighted_metrics = []
        weighted_metrics = [tf.keras.metrics.BinaryCrossentropy(from_logits=True),
                            tf.keras.metrics.BinaryAccuracy(threshold=0)] + weighted_metrics
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        super().compile(loss=loss, weighted_metrics=weighted_metrics, **kwargs)

    def evaluate(self, x=None, return_dict=False, **kwargs):
        # If the dataset is empty, the evaluation terminates gracefully.
        # Model.evaluate throws OverflowError in such case.
        try:
            next(iter(x))
        except StopIteration:
            if return_dict:
                return {}
            else:
                return []
        return super().evaluate(x=x, return_dict=return_dict, **kwargs)

    @staticmethod
    def raggify(x):
        if isinstance(x, dict):
            flat_values = x['flat_values']
            nested_row_splits = x['nested_row_splits']
            if len(flat_values.shape) == 2:
                flat_values = tf.squeeze(flat_values, axis=1)
                nested_row_splits = tuple(tf.squeeze(e, axis=1) for e in nested_row_splits)
            x = tf.RaggedTensor.from_nested_row_splits(flat_values, nested_row_splits)
        return x

    def reset_metrics(self):
        super().reset_metrics()
        self.symbol_weight_model.reset_metrics()

    def test_step(self, x):
        x, y, sample_weight = x

        # Updates stateful loss metrics.
        # TODO: Collect per-sample loss values across an epoch and then log their histogram.
        y_pred, loss = self.compute_loss(x, y, sample_weight)

        return self.update_metrics(y, y_pred, sample_weight)

    def train_step(self, x):
        x, y, sample_weight = x
        with tf.GradientTape() as tape:
            y_pred, loss = self.compute_loss(x, y, sample_weight)
        self.optimizer_minimize(loss, self.trainable_variables, tape)
        return self.update_metrics(y, y_pred, sample_weight)

    def compute_loss(self, x, y, sample_weight):
        # Sample weights are only supported with flat (non-ragged) tensors.
        flat_y = tf.expand_dims(y.flat_values, 1)
        flat_sample_weight = tf.expand_dims(sample_weight.flat_values, 1)

        n_problems = tf.shape(y.row_splits)[0] - 1
        n_problems = tf.cast(n_problems, sample_weight.dtype)
        # tf.debugging.assert_near(tf.reduce_sum(sample_weight), n_problems, rtol=tf.keras.backend.epsilon() * len(y.flat_values))

        # TODO: Make sure that the loss does not depend on the batch size.
        y_pred = self(x, training=True)
        flat_y_pred = tf.expand_dims(y_pred.flat_values, 1)
        return y_pred, self.compiled_loss(flat_y, flat_y_pred, flat_sample_weight / n_problems,
                                          regularization_losses=self.losses)

    def optimizer_minimize(self, loss, var_list, tape):
        # Inspiration:
        # self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        # https://stackoverflow.com/a/62754676
        grads_and_vars = self.optimizer._compute_gradients(loss, var_list=var_list, tape=tape)
        return self.optimizer.apply_gradients((g, v) for g, v in grads_and_vars if g is not None)

    def update_metrics(self, y, y_pred, sample_weight):
        flat_y = tf.expand_dims(y.flat_values, 1)
        flat_y_pred = tf.expand_dims(y_pred.flat_values, 1)
        flat_y_sample_weight = tf.expand_dims(sample_weight.flat_values, 1)
        self.compiled_metrics.update_state(flat_y, flat_y_pred, flat_y_sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x, training=False):
        problem = x['problem']
        occurrence_count = self.raggify(x['occurrence_count'])
        token_weight_decorated = self.symbol_weight_model(problem, training=training)
        logits = self.costs_decorated_to_logits(token_weight_decorated, occurrence_count)
        return logits

    @classmethod
    @tf.function
    def costs_decorated_to_logits(cls, symbol_costs_decorated, questions):
        symbol_costs_valid = tf.ragged.boolean_mask(symbol_costs_decorated['costs'], symbol_costs_decorated['valid'])
        questions_valid = tf.ragged.boolean_mask(questions, symbol_costs_decorated['valid'])
        logits_valid = cls.costs_to_logits(symbol_costs_valid, questions_valid)
        index_mask = tf.repeat(symbol_costs_decorated['valid'], questions.row_lengths())
        indices_all = tf.range(questions.row_splits[-1])
        indices_valid = indices_all[index_mask]
        indices_valid = tf.expand_dims(indices_valid, axis=1)
        # tf.debugging.assert_equal(indices_valid.shape[:-1], logits_valid.flat_values.shape)
        # We assign the value 0 to the invalid logits. `scatter_nd` defaults the output values to 0.
        logits_values = tf.scatter_nd(indices_valid, logits_valid.flat_values,
                                      tf.reshape(questions.row_splits[-1], (1,)))
        logits = tf.RaggedTensor.from_row_splits(logits_values, questions.row_splits)
        tf.debugging.assert_equal(logits.row_splits, questions.row_splits)
        return logits

    @classmethod
    @tf.function
    def costs_to_logits(cls, token_weight, occurrence_count):
        """
        :param token_weight: Shape: `[num_problems, (num_symbols)]`
        :param occurrence_count: Shape: `[num_problems, (num_questions), (num_symbols)]`
        """
        tf.debugging.assert_equal(token_weight.nrows(), occurrence_count.nrows())
        sc_like_questions = cls.tile_like(token_weight, occurrence_count)
        tf.debugging.assert_equal(occurrence_count.nested_row_splits[0], sc_like_questions.nested_row_splits[0])
        # This assertion fails if there is a mismatch in signature sizes, namely if there is a mismatch of symbol type.
        tf.debugging.assert_equal(occurrence_count.nested_row_splits[1], sc_like_questions.nested_row_splits[1])
        # tf.debugging.assert_greater_equal(2 / (tf.cast(token_weight.row_lengths(), occurrence_count.dtype) + 1),
        #                                  tf.reduce_max(occurrence_count, axis=(1, 2)))
        # tf.debugging.assert_less_equal(-2 / (tf.cast(token_weight.row_lengths(), occurrence_count.dtype) + 1),
        #                               tf.reduce_min(occurrence_count, axis=(1, 2)))
        potentials = tf.ragged.map_flat_values(tf.multiply, occurrence_count, sc_like_questions)
        logits = tf.reduce_sum(potentials, axis=2)
        # assert (logits.shape + (None,)).as_list() == questions.shape.as_list()
        # assert tf.reduce_all(logits.row_splits == questions.row_splits)
        return logits

    @staticmethod
    @tf.function
    def tile_like(symbol_costs, questions):
        tf.debugging.assert_equal(symbol_costs.nrows(), questions.nrows())
        n_questions = questions.row_lengths()
        starts = tf.repeat(symbol_costs.row_splits[:-1], n_questions)
        limits = tf.repeat(symbol_costs.row_splits[1:], n_questions)
        indices = tf.ragged.range(starts, limits)
        # Shape: `[num_questions, (num_symbols)]` where `num_questions` is the total number of questions.
        sc_tiled = tf.gather(symbol_costs.flat_values, indices)
        sc_like_questions = tf.RaggedTensor.from_row_splits(sc_tiled, questions.row_splits)
        return sc_like_questions
