import dgl.nn.tensorflow as dglnn
import tensorflow as tf

from proving.formula_visitor import FormulaVisitor

dtype_tf_float = tf.float32


class GCN(tf.keras.layers.Layer):
    def __init__(self, canonical_etypes, ntype_in_degrees, ntype_feat_sizes, args, output_ntypes=None, constraint=None,
                 name='gcn', **kwargs):
        super().__init__(name=name, **kwargs)

        activation = args.activation
        if isinstance(activation, str):
            activation = tf.keras.activations.deserialize(activation)

        ntypes = ntype_in_degrees.keys()

        if args.aggregate == 'concat':
            ntype_embedding_lengths = {ntype: in_degree * args.message_size for ntype, in_degree in
                                       ntype_in_degrees.items()}

            def aggregate_fn(tensors, dsttype):
                return tf.concat(tensors, axis=1)
        else:
            ntype_embedding_lengths = {ntype: args.message_size for ntype in ntypes}
            aggregate_fn = args.aggregate

        self.ntype_embeddings = {
            ntype: self.add_ntype_embedding(ntype, ntype_embedding_lengths[ntype], ntype_feat_sizes.get(ntype, 0),
                                            constraint, args.dropout.input) for ntype in ntypes}

        stype_feats = ntype_embedding_lengths
        dtype_feats = {ntype: args.message_size for ntype in ntypes}

        def create_module(in_feats, out_feats, norm, name):
            # We assume that there are no 0-in-degree nodes in any input graph.
            # This holds for the standard graphification scheme because all symbols have loops and all the other nodes
            # have at least one in-edge from another node.
            return GraphConv(in_feats, out_feats, norm=norm, dropout=args.dropout.hidden, constraint=constraint,
                             name=name, activation=activation, allow_zero_in_degree=True)

        layer_norm_ntypes = None
        layers_reversed = []
        if output_ntypes is None:
            output_ntypes = ntypes
        contributing_dtypes = set(output_ntypes)
        for layer_i in range(args.depth - 1, -1, -1):
            contributing_stypes = set()
            mods = {}
            for stype, etype, dtype in canonical_etypes:
                # Only add the module if it transitively contributes to `output_ntypes` at the last layer.
                if dtype in contributing_dtypes:
                    contributing_stypes.add(stype)
                    if etype not in mods:
                        if args.conv_norm is not None:
                            norm = args.conv_norm
                        else:
                            norm = FormulaVisitor.conv_norm(etype)
                        mods[etype] = create_module(stype_feats[stype], dtype_feats[dtype], norm,
                                                    f'layer_{layer_i}/{etype}')
            if args.layer_norm:
                layer_norm_ntypes = contributing_dtypes
            layers_reversed.append(HeteroGraphConv(mods, residual=args.residual, layer_norm_ntypes=layer_norm_ntypes,
                                                   aggregate=aggregate_fn, name=f'layer_{layer_i}'))
            # Update stype_feats
            if args.aggregate == 'concat':
                stype_feats = {ntype: 0 for ntype in ntypes}
                for stype, etype, dtype in canonical_etypes:
                    stype_feats[dtype] += dtype_feats[stype]
            else:
                stype_feats = dtype_feats
            contributing_dtypes = contributing_stypes
        self.layers = list(reversed(layers_reversed))

    def call(self, g, training=False):
        h = self.initial_h(g, training=training)
        for layer in self.layers:
            h = layer(g, h, training=training)
        return h

    def initial_h(self, graph, training=False):
        return {ntype: self.initial_ntype_embedding(graph, ntype, training=training) for ntype in graph.ntypes}

    def initial_ntype_embedding(self, graph, ntype, training=False):
        return self.ntype_embeddings[ntype].embedding(graph.num_nodes(ntype), graph.ndata['feat'].get(ntype),
                                                      training=training)

    def add_ntype_embedding(self, ntype, embedding_size, feat_size, constraint, dropout):
        weight = self.add_ntype_weight(ntype, embedding_size - feat_size, constraint)
        return TrainableEmbedding(weight, feat_size, dropout)

    def add_ntype_weight(self, name, size, constraint):
        # We only use flat embeddings because `GraphConv` only supports flat input.
        return self.add_weight(name=name, shape=(1, size), initializer='random_normal', constraint=constraint,
                               trainable=True)


class TrainableEmbedding:
    def __init__(self, weight, feat_size=0, dropout=None):
        self.weight = weight
        self.feat_size = feat_size
        self.dropout = None
        if dropout is not None:
            self.dropout = tf.keras.layers.Dropout(dropout)

    def embedding(self, n, feat=None, training=False):
        embedding = tf.tile(self.weight, (n, 1))
        if self.dropout is not None:
            embedding = self.dropout(embedding, training=training)
        assert feat is None or feat.shape == (n, self.feat_size)
        if self.feat_size > 0:
            if feat is None:
                feat = tf.zeros((n, self.feat_size))
            embedding = tf.concat((feat, embedding), axis=1)
        return embedding


class HeteroGraphConv(dglnn.HeteroGraphConv):
    def __init__(self, mods, residual=True, layer_norm_ntypes=None, name=None, **kwargs):
        # High-level architecture:
        # See slide 27 in https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/slides.pdf/npfl114-07.pdf
        # 1. Fork residual
        # 2. `GraphConv`
        #     1. Dropout
        #     2. Dense layer
        # 3. Aggregate across source node types
        # 4. Join residual
        # 5. Layer normalization
        super().__init__(mods, **kwargs)
        self.residual = residual
        if layer_norm_ntypes is None:
            layer_norm_ntypes = []
        # We need not train scaling because a fully connected layer follows.
        self.layer_norm = {ntype: tf.keras.layers.LayerNormalization(scale=False, name=ntype) for ntype in
                           layer_norm_ntypes}
        if name is not None:
            self._name = name

    def __repr__(self):
        return f'{type(self).__name__}(name={self.name}, mods=[{len(self.mods)}], residual={self.residual}, layer_norm=[{len(self.layer_norm)}])'

    def call(self, g, inputs, training=False, mod_kwargs=None, **kwargs):
        if mod_kwargs is None:
            mod_kwargs = {}
        mod_kwargs['training'] = training
        outputs = self.super_call(g, inputs, mod_kwargs=mod_kwargs, **kwargs)
        if self.residual:
            assert set(outputs) <= set(inputs)
            outputs = {k: inputs[k] + outputs[k] for k in outputs}
        for k, layer in self.layer_norm.items():
            if outputs[k].shape[0] >= 1:
                outputs[k] = layer(outputs[k], training=training)
        return outputs

    def super_call(self, g, inputs, mod_args=None, mod_kwargs=None):
        # Slight modification of `dglnn.HeteroGraphConv.call`: Calls the sub-module even if number of edges is 0.
        outputs = self.apply_mods(g, inputs, mod_args, mod_kwargs)
        rsts = self.aggregate(outputs)
        return rsts

    def apply_mods(self, g, inputs, mod_args, mod_kwargs):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        for stype, etype, dtype in g.canonical_etypes:
            if etype not in self.mods or stype not in inputs:
                continue
            rel_graph = g[stype, etype, dtype]
            dstdata = self.mods[etype](
                rel_graph,
                inputs[stype],
                *mod_args.get(etype, ()),
                **mod_kwargs.get(etype, {}))
            outputs[dtype].append(dstdata)
        return outputs

    def aggregate(self, outputs):
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts


class GraphConv(dglnn.GraphConv):
    def __init__(self, in_feats, out_feats, dropout=None, constraint=None, name=None, weight=True, **kwargs):
        super().__init__(in_feats, out_feats, weight=False, **kwargs)
        self.dropout = None
        if dropout is not None:
            self.dropout = tf.keras.layers.Dropout(dropout)
        # `self.weight` initialization mimics `dglnn.GraphConv.__init__`.
        # The difference is that here we add a MaxNorm constraint.
        if weight:
            xinit = tf.keras.initializers.glorot_uniform()
            self.weight = tf.Variable(initial_value=xinit(shape=(in_feats, out_feats), dtype='float32'), trainable=True,
                                      constraint=constraint)
        else:
            self.weight = None
        if name is not None:
            self._name = name

    def call(self, graph, feat, training=False):
        # `feat` dimensions:
        # - 0: source nodes (across multiple problems in the batch)
        # - 1: source node embedding channels
        # It doesn't make sense to use batch normalization namely because each problem contributes a different number of nodes.
        if graph.number_of_edges() == 0:
            # There are no edges from stype to dtype.
            # The message is formed by tiling the bias, which corresponds to summation over 0 inputs.
            assert len(graph.dsttypes) == 1
            dtype = graph.dsttypes[0]
            outputs = tf.tile(tf.expand_dims(tf.stop_gradient(self.bias), 0), (graph.num_nodes(dtype), 1))
        else:
            # `dglnn.GraphConv` does not accept the argument `training`.
            outputs = super().call(graph, feat)
        if self.dropout is not None:
            outputs = self.dropout(outputs, training=training)
        return outputs
