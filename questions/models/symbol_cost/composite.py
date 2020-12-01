import tensorflow as tf

from .symbol_cost import SymbolCostModel


class Composite(SymbolCostModel):
    def __init__(self, problem_to_embedding, embedding_to_cost=None):
        super().__init__()
        self.problem_to_embedding = problem_to_embedding
        if embedding_to_cost is None:
            embedding_to_cost = tf.keras.layers.Dense(1, name='embedding_to_cost')
        self.embedding_to_cost = embedding_to_cost
        self.symbol_cost_metrics = []

    def call(self, problems, training=False):
        # `Model.test_step` pads `problems` with a length 1 axis.
        if len(problems.shape) == 2:
            problems = tf.squeeze(problems, axis=1)
        embeddings = self.problem_to_embedding(problems, training=training)
        # `TimeDistributed` can be used in eager mode to support ragged tensor input.
        # In non-eager mode it is necessary to flatten the ragged tensor `embedding`.
        costs_flat_values = self.embedding_to_cost(embeddings['embeddings'].flat_values, training=training)
        costs_flat_values = tf.squeeze(costs_flat_values, axis=1)
        costs = tf.RaggedTensor.from_nested_row_splits(costs_flat_values, embeddings['embeddings'].nested_row_splits)
        return {'costs': costs, 'valid': embeddings['valid']}
