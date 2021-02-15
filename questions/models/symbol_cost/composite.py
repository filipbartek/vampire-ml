import tensorflow as tf

from .symbol_cost import SymbolCostModel


class Composite(SymbolCostModel):
    def __init__(self, problem_to_embedding, embedding_to_cost, l2=0):
        super().__init__()
        self.problem_to_embedding = problem_to_embedding
        self.embedding_to_cost = embedding_to_cost
        self.l2 = l2
        self.symbol_cost_metrics = []

    def call(self, problems, training=False, cache=True):
        # `Model.test_step` pads `problems` with a length 1 axis.
        if len(problems.shape) == 2:
            problems = tf.squeeze(problems, axis=1)
        embeddings = self.problem_to_embedding(problems, training=training, cache=cache)
        # `TimeDistributed` can be used in eager mode to support ragged tensor input.
        # In non-eager mode it is necessary to flatten the ragged tensor `embedding`.
        costs_flat_values = self.embedding_to_cost(embeddings['embeddings'].flat_values, training=training)
        costs_flat_values = tf.squeeze(costs_flat_values, axis=1)
        costs = tf.RaggedTensor.from_nested_row_splits(costs_flat_values, embeddings['embeddings'].nested_row_splits)
        # For each problem, its regularization loss is proportional to the l2-norm of the symbol cost vector.
        # More precisely, it is `reduce_mean(square(x))` where x is the symbol cost vector.
        # We take the mean over symbols instead of sum so that problems with large signature don't contribute more than problems with small signature.
        # We aggregate across problems by taking the mean.
        valid_costs = tf.ragged.boolean_mask(costs, embeddings['valid'])
        self.add_loss(self.l2_loss(valid_costs))
        return {'costs': costs, 'valid': embeddings['valid']}

    @tf.function(experimental_relax_shapes=True)
    def l2_loss(self, symbol_costs):
        if symbol_costs.nrows() >= 1:
            return self.l2 * tf.reduce_mean(tf.reduce_mean(tf.square(symbol_costs), axis=1))
        else:
            return tf.constant(0, dtype=symbol_costs.dtype)
