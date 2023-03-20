import tensorflow as tf

from .symbol_cost import SymbolCostModel


class Composite(SymbolCostModel):
    def __init__(self, problem_to_embedding, embedding_to_cost, l2=0):
        super().__init__()
        self.problem_to_embedding = problem_to_embedding
        self.embedding_to_cost = embedding_to_cost
        self.l2 = l2
        self.symbol_cost_metrics = []

    def call(self, problems, training=False, cache=None):
        # `Model.test_step` pads `problems` with a length 1 axis.
        if len(problems.shape) == 2:
            problems = tf.squeeze(problems, axis=1)
        embeddings = self.problem_to_embedding(problems, training=training, cache=cache)

        clause_feature_to_node_type = {
            'symbol': 'symbol',
            'literal_positive': 'atom',
            'literal_negative': 'atom',
            'equality': 'equality',
            'inequality': 'equality',
            'variable_occurrence': 'variable',
            'number': 'function'
        }
        costs_dict = {k: etc(embeddings['embeddings'][clause_feature_to_node_type[k]], training=training) for k, etc in
                      self.embedding_to_cost.items()}
        costs_dict['symbol'] = tf.squeeze(costs_dict['symbol'], axis=-1)
        costs = tf.concat(list(costs_dict.values()), axis=1)
        # For each problem, its regularization loss is proportional to the l2-norm of the symbol cost vector.
        # More precisely, it is `reduce_mean(square(x))` where x is the symbol cost vector.
        # We take the mean over symbols instead of sum so that problems with large signature don't contribute more than problems with small signature.
        # We aggregate across problems by taking the mean.
        valid_costs = tf.ragged.boolean_mask(costs, embeddings['valid'])
        self.add_loss(self.l2_loss(valid_costs))
        return {'costs': costs, 'valid': embeddings['valid']}

    # TODO: Add function call_one that returns some logs about the processing, namely clausification time.

    @tf.function(experimental_relax_shapes=True)
    def l2_loss(self, symbol_costs):
        if symbol_costs.nrows() >= 1:
            return self.l2 * tf.reduce_mean(tf.reduce_mean(tf.square(symbol_costs), axis=1))
        else:
            return tf.constant(0, dtype=symbol_costs.dtype)
