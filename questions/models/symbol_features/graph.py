import dgl
import tensorflow as tf

from questions.utils import py_str
from .symbol_features import SymbolFeatures


class Graph(SymbolFeatures):
    def __init__(self, graphifier, gcn, common_clause_feature_count=0):
        SymbolFeatures.__init__(self, dynamic=True)
        self.gcn = gcn
        self.graphifier = graphifier
        self.special_token_model = self.add_weight(name='special_token_weight',
                                                   shape=(1, common_clause_feature_count, 16),
                                                   initializer='random_normal', trainable=True)

    def call(self, problems, training=False, cache=True):
        batch_graph, valid = self.problems_to_batch_graph(problems, cache=cache)
        res = self.resolve_batch_graph(batch_graph, training=training)
        return {'embeddings': res, 'valid': valid}

    def problems_to_batch_graph(self, problems, cache):
        with tf.device('/cpu'):
            problems = list(map(py_str, problems))
            graphs = self.graphifier.get_graphs(problems, cache=cache, get_df=False)

            def convert(g):
                if g is None:
                    return self.graphifier.empty_graph(), False
                else:
                    return g, True

            graphs, valid = zip(*map(convert, graphs))
            batch_graph = dgl.batch(graphs)
            valid = tf.convert_to_tensor(valid, dtype=tf.bool)
        valid = tf.identity(valid)
        batch_graph = batch_graph.to(valid.device)
        return batch_graph, valid

    def resolve_batch_graph(self, batch_graph, training=False):
        values = self.gcn(batch_graph, training=training)
        row_lengths = {k: batch_graph.batch_num_nodes(k) for k in values}
        ragged_tensors = {k: tf.RaggedTensor.from_row_lengths(v, row_lengths[k]) for k, v in values.items()}
        # TODO: Replace the dummy embeddings with some output of the GCN.
        special_token_embeddings = tf.tile(self.special_token_model, [batch_graph.batch_size, 1, 1])
        # TODO: Make sure the concatenation happens on GPU even with ragged tensors.
        res = tf.concat([special_token_embeddings, ragged_tensors['predicate'], ragged_tensors['function']], 1)
        return res
