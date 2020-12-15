import dgl
import tensorflow as tf

from proving.utils import py_str
from .heterographconv import HeteroGCN
from .symbol_features import SymbolFeatures


class Graph(SymbolFeatures, HeteroGCN):
    def __init__(self, graphifier, graphs, symbol_type, edge_layer_sizes=64, node_layer_sizes=64, num_layers=1,
                 activation='relu', dropout=0, kernel_max_norm=None):
        SymbolFeatures.__init__(self, dynamic=True)
        if isinstance(edge_layer_sizes, int):
            edge_layer_sizes = {canonical_etype: edge_layer_sizes for canonical_etype in graphifier.canonical_etypes}
        if isinstance(node_layer_sizes, int):
            node_layer_sizes = {ntype: node_layer_sizes for ntype in graphifier.ntypes}
        HeteroGCN.__init__(self, edge_layer_sizes, node_layer_sizes, num_layers, output_ntypes=[symbol_type],
                           dynamic=True, activation=activation, dropout=dropout, kernel_max_norm=kernel_max_norm)
        self.graphifier = graphifier
        self.graphs = graphs
        self.symbol_type = symbol_type

    def call(self, problems):
        batch_graph, valid = self.problems_to_batch_graph(problems)
        res = self.resolve_batch_graph(batch_graph)
        return {'embeddings': res, 'valid': valid}

    def problems_to_batch_graph(self, problems):
        def gen():
            for p in problems:
                try:
                    yield self.graphs[py_str(p)], True
                except KeyError:
                    yield self.graphifier.empty_graph(), False

        graphs, valid = zip(*gen())
        batch_graph = dgl.batch(graphs)
        valid = tf.convert_to_tensor(valid, dtype=tf.bool)
        return batch_graph, valid

    def resolve_batch_graph(self, batch_graph):
        values = HeteroGCN.call(self, batch_graph)[self.symbol_type]
        row_lengths = batch_graph.batch_num_nodes(self.symbol_type)
        ragged_tensor = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        return ragged_tensor
