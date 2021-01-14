import dgl.nn.tensorflow as dglnn
import tensorflow as tf

dtype_tf_float = tf.float32


class GCN(tf.keras.layers.Layer):
    def __init__(self, canonical_etypes, ntype_in_degrees, ntype_feat_sizes, embedding_size=64, depth=4,
                 aggregate='concat', activation='relu', dropout=None, name='gcn', **kwargs):
        super().__init__(name=name, **kwargs)

        if isinstance(activation, str):
            activation = tf.keras.activations.deserialize(activation)

        ntypes = ntype_in_degrees.keys()

        if aggregate == 'concat':
            ntype_embedding_lengths = {ntype: in_degree * embedding_size for ntype, in_degree in
                                       ntype_in_degrees.items()}

            def aggregate_fn(tensors, dsttype):
                return tf.concat(tensors, axis=1)
        else:
            ntype_embedding_lengths = {ntype: embedding_size for ntype in ntypes}
            aggregate_fn = aggregate

        def ntype_weight_length(ntype):
            assert ntype in ntype_embedding_lengths
            if ntype in ntype_feat_sizes:
                assert ntype_embedding_lengths[ntype] >= ntype_feat_sizes[ntype]
                return ntype_embedding_lengths[ntype] - ntype_feat_sizes[ntype]
            return ntype_embedding_lengths[ntype]

        ntype_weight_lengths = {ntype: ntype_weight_length(ntype) for ntype in ntypes}
        self.ntype_weights = self.add_ntype_weights(ntype_weight_lengths)
        self.ntype_feat_sizes = ntype_feat_sizes

        stype_feats = ntype_embedding_lengths
        dtype_feats = {ntype: embedding_size for ntype in ntypes}

        def create_module(in_feats, out_feats, name):
            return GraphConv(in_feats, out_feats, dropout=dropout, name=name, activation=activation,
                             allow_zero_in_degree=True)

        self.layers = []
        for layer_i in range(depth):
            mods = {etype: create_module(stype_feats[stype], dtype_feats[dtype], f'layer_{layer_i}/{etype}') for
                    stype, etype, dtype in canonical_etypes}
            self.layers.append(dglnn.HeteroGraphConv(mods, aggregate=aggregate_fn))
            # Update stype_feats
            if aggregate == 'concat':
                stype_feats = {ntype: 0 for ntype in ntypes}
                for stype, etype, dtype in canonical_etypes:
                    stype_feats[dtype] += dtype_feats[stype]
            else:
                stype_feats = dtype_feats

    def call(self, g, training=False):
        h = self.initial_h(g)
        for layer in self.layers:
            h = layer(g, h, training=training)
        return h

    def initial_h(self, graph):
        return {ntype: self.initial_ntype_embedding(graph, ntype) for ntype in graph.ntypes}

    def initial_ntype_embedding(self, graph, ntype):
        embedding = tf.tile(self.ntype_weights[ntype], (graph.num_nodes(ntype), 1))
        if ntype in graph.ndata['feat']:
            # Prepend the node features to the trainable embedding
            embedding = tf.concat((graph.ndata['feat'][ntype], embedding), axis=1)
        return embedding

    def add_ntype_weights(self, ntype_weight_lengths):
        return {ntype: self.add_ntype_weight(ntype, l) for ntype, l in ntype_weight_lengths.items()}

    def add_ntype_weight(self, name, size):
        # We only use flat embeddings because `GraphConv` only supports flat input.
        return self.add_weight(name=name, shape=(1, size), initializer='random_normal', trainable=True)


class GraphConv(dglnn.GraphConv):
    def __init__(self, in_feats, out_feats, dropout=None, name=None, **kwargs):
        super().__init__(in_feats, out_feats, **kwargs)
        self.dropout = None
        if dropout is not None:
            self.dropout = tf.keras.layers.Dropout(dropout)
        if name is not None:
            self._name = name

    def call(self, graph, feat, training=False):
        if self.dropout is not None:
            feat = self.dropout(feat, training=training)
        # `dglnn.GraphConv` does not accept the argument `training`.
        feat = super().call(graph, feat)
        return feat
