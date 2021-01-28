import dgl
import tensorflow as tf

from proving.utils import py_str
from .symbol_features import SymbolFeatures


class Graph(SymbolFeatures):
    def __init__(self, empty_graph, graphs, symbol_type, gcn):
        SymbolFeatures.__init__(self, dynamic=True)
        self.gcn = gcn
        self.empty_graph = empty_graph
        self.graphs = graphs
        self.symbol_type = symbol_type

    def call(self, problems, training=False):
        batch_graph, valid = self.problems_to_batch_graph(problems)
        res = self.resolve_batch_graph(batch_graph, training=training)
        return {'embeddings': res, 'valid': valid}

    def problems_to_batch_graph(self, problems):
        def gen():
            for p in problems:
                try:
                    yield self.graphs[py_str(p)], True
                except KeyError:
                    yield self.empty_graph, False

        graphs, valid = zip(*gen())
        batch_graph = dgl.batch(graphs)
        valid = tf.convert_to_tensor(valid, dtype=tf.bool)
        return batch_graph, valid

    def resolve_batch_graph(self, batch_graph, training=False):
        values = self.gcn(batch_graph, training=training)[self.symbol_type]
        row_lengths = batch_graph.batch_num_nodes(self.symbol_type)
        ragged_tensor = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        return ragged_tensor
