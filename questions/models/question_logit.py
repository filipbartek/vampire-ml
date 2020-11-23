import tensorflow as tf


class QuestionLogitModel(tf.keras.Model):
    def __init__(self, symbol_cost_model, update_symbol_cost_metrics=False):
        super().__init__()
        self.symbol_cost_model = symbol_cost_model
        self.update_symbol_cost_metrics = update_symbol_cost_metrics

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

    @staticmethod
    def get_sample_weight(questions):
        # This sample weight vector ensures that each problem has the same weight.
        return 1 / tf.repeat(questions.row_lengths(), questions.row_lengths())

    @classmethod
    def get_loss_sample_weight(cls, questions):
        # We want the loss to be the mean of the batch: https://stats.stackexchange.com/a/358971/271804
        # This means the mean over problems, not questions.
        # The loss sums (not averages) the contributions from questions,
        # weighted so that each problem contributes a unit.
        # We normalize the loss by dividing by the number of problems.
        sample_weight = cls.get_sample_weight(questions)
        n_problems = tf.cast(questions.shape[0], sample_weight.dtype)
        tf.debugging.assert_near(tf.reduce_sum(sample_weight), n_problems)
        return sample_weight / n_problems
    def test_step(self, x):
        # We assume that only x is passed, not y nor sample_weight.
        questions = self.raggify_questions(x['questions'])
        y = tf.zeros((questions.row_splits[-1], 1))
        x['questions'] = questions

        y_pred = tf.reshape(self(x, training=False).flat_values, (-1, 1))

        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, self.get_loss_sample_weight(questions), regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, self.get_sample_weight(questions))

        res = {m.name: m.result() for m in self.metrics}

        if self.update_symbol_cost_metrics and self.symbol_cost_model.compiled_metrics is not None:
            symbol_cost_results = self.symbol_cost_model.test_step(x['problems'])
            res.update({k: v for k, v in symbol_cost_results.items() if k != 'loss'})

        return res

    def train_step(self, x):
        questions = self.raggify_questions(x['questions'])
        y = tf.zeros((questions.row_splits[-1], 1))
        x['questions'] = questions

        with tf.GradientTape() as tape:
            y_pred = tf.reshape(self(x, training=False).flat_values, (-1, 1))
            loss = self.compiled_loss(y, y_pred, self.get_loss_sample_weight(questions),
                                      regularization_losses=self.losses)
        tf.python.keras.engine.training._minimize(self.distribute_strategy, tape, self.optimizer, loss,
                                                  self.trainable_variables)

        self.compiled_metrics.update_state(y, y_pred, self.get_sample_weight(questions))

        res = {m.name: m.result() for m in self.metrics}

        if self.update_symbol_cost_metrics and self.symbol_cost_model.compiled_metrics is not None:
            symbol_cost_results = self.symbol_cost_model.test_step(x['problems'])
            res.update({k: v for k, v in symbol_cost_results.items() if k != 'loss'})

        return res

    def call(self, x, training=False):
        problems = x['problems']
        questions = self.raggify_questions(x['questions'])
        symbol_costs_decorated = self.symbol_cost_model(problems, training=training)
        logits = self.costs_decorated_to_logits(symbol_costs_decorated, questions)
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
        logits_values = tf.scatter_nd(indices_valid, logits_valid.flat_values,
                                      tf.reshape(questions.row_splits[-1], (1,)))
        logits = tf.RaggedTensor.from_row_splits(logits_values, questions.row_splits)
        tf.debugging.assert_equal(logits.row_splits, questions.row_splits)
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
        if start is None and step is not None:
            start = 0
        elif step is None and start is not None:
            step = 1
        self.start = start
        self.step = step

    def on_epoch_end(self, epoch, logs=None):
        if self.start is not None and self.step is not None and epoch >= self.start and (
                epoch - self.start) % self.step == 0:
            assert logs is not None
            symbol_cost_model = self.model.symbol_cost_model
            if self.problems is not None and self.problems.cardinality() >= 1:
                print(f'Evaluating symbol cost model on {self.problems.cardinality()} training problems...')
                train_res = symbol_cost_model.evaluate(self.problems, return_dict=True)
                logs.update({k: v for k, v in train_res.items() if k != 'loss'})
            if self.problems_validation is not None and self.problems_validation.cardinality() >= 1:
                print(
                    f'Evaluating symbol cost model on {self.problems_validation.cardinality()} validation problems...')
                validation_res = symbol_cost_model.evaluate(self.problems_validation, return_dict=True)
                logs.update({f'val_{k}': v for k, v in validation_res.items() if k != 'loss'})
            print(f'Metrics after epoch {epoch}: {logs}')
