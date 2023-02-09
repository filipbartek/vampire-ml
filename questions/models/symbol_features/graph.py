import dgl
import math
import tensorflow as tf

from questions.utils import py_str
from .symbol_features import SymbolFeatures


class Graph(SymbolFeatures):
    def __init__(self, graphifier, gcn, readout_op='sum'):
        SymbolFeatures.__init__(self, dynamic=True)
        self.gcn = gcn
        self.graphifier = graphifier
        self.readout_op = readout_op

    def call(self, problems, training=False, expensive=True):
        batch_graph, valid = self.problems_to_batch_graph(problems, training=training, expensive=expensive)
        res = self.resolve_batch_graph(batch_graph, training=training)
        return {'embeddings': res, 'valid': valid}

    def problems_to_batch_graph(self, problems, training, expensive):
        with tf.device('/cpu'):
            problems = list(map(py_str, problems))
            graphs = self.graphifier.get_graphs(problems, training=training, expensive=expensive, get_df=False)

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

        symbol_ntypes = ['predicate', 'function']
        assert set(symbol_ntypes) <= set(values)
        row_lengths = {k: batch_graph.batch_num_nodes(k) for k in symbol_ntypes}
        ragged_tensors = {k: tf.RaggedTensor.from_row_lengths(values[k], row_lengths[k]) for k in symbol_ntypes}

        readout_ntypes = ['variable', 'atom', 'equality', 'function']
        assert set(readout_ntypes) <= set(values)
        with batch_graph.local_scope():
            batch_graph.ndata['h'] = {k: values[k] for k in readout_ntypes}
            readouts = {ntype: dgl.readout_nodes(batch_graph, 'h', ntype=ntype, op=self.readout_op) for ntype in
                        readout_ntypes}

        # The readout values are all zeros for graphs and ntypes such that there is no ntype node in the graph.
        # Replace such zeros with nans.
        #for ntype in readouts:
        #    num_nodes = batch_graph._batch_num_nodes[ntype]
        #    mask_nan = tf.tile(tf.expand_dims(num_nodes == 0, 1), (1, 16))
        #    readouts[ntype] = tf.where(mask_nan, math.nan, readouts[ntype])

        res = {
            **readouts,
            # TODO: Make sure the concatenation happens on GPU even with ragged tensors.
            'symbol': tf.concat(list(ragged_tensors.values()), 1)
        }
        return res
