import dgl
import tensorflow as tf

from proving.utils import py_str
from .symbol_features import SymbolFeatures


class Graph(SymbolFeatures):
    def __init__(self, graphifier, symbol_type, gcn):
        SymbolFeatures.__init__(self, dynamic=True)
        self.gcn = gcn
        self.graphifier = graphifier
        self.symbol_type = symbol_type

    def call(self, problems, training=False):
        batch_graph, valid = self.problems_to_batch_graph(problems)
        res = self.resolve_batch_graph(batch_graph, training=training)
        return {'embeddings': res, 'valid': valid}

    def problems_to_batch_graph(self, problems):
        problems = list(map(py_str, problems))
        graphs = self.graphifier.get_graphs(problems, get_df=False)

        def convert(g):
            if g is None:
                return self.graphifier.empty_graph(), False
            else:
                return g, True

        graphs, valid = zip(*map(convert, graphs))
        batch_graph = dgl.batch(graphs)
        valid = tf.convert_to_tensor(valid, dtype=tf.bool)
        return batch_graph, valid

    def resolve_batch_graph(self, batch_graph, training=False):
        values = self.gcn(batch_graph, training=training)[self.symbol_type]
        row_lengths = batch_graph.batch_num_nodes(self.symbol_type)
        ragged_tensor = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        return ragged_tensor
