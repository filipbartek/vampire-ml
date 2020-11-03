import functools

import dgl
import tensorflow as tf

dtype_tf_float = tf.float32


class HeteroGCN(tf.keras.layers.Layer):
    def __init__(self, edge_layer_sizes, node_layer_sizes, num_layers=1, output_ntypes=None):
        super().__init__()

        assert num_layers >= 1

        if output_ntypes is None:
            output_ntypes = node_layer_sizes.keys()
        self.output_ntypes = output_ntypes

        layers_list_reversed = []
        contributing_srctypes = set()
        contributing_dsttypes = set(output_ntypes)
        for i in range(num_layers):
            # Every node contributes to itself.
            contributing_srctypes = contributing_dsttypes.copy()
            for srctype, etype, dsttype in edge_layer_sizes.keys():
                if dsttype in contributing_dsttypes:
                    contributing_srctypes.add(srctype)
            cur_edge_layer_sizes = {k: v for k, v in edge_layer_sizes.items() if
                                    k[0] in contributing_srctypes and k[2] in contributing_dsttypes}
            cur_node_layer_sizes = {k: node_layer_sizes[k] for k in contributing_dsttypes}
            layers_list_reversed.append(self.create_layer(cur_edge_layer_sizes, cur_node_layer_sizes))
            contributing_dsttypes = contributing_srctypes
        self.layers_list = list(reversed(layers_list_reversed))

        assert contributing_srctypes <= set(node_layer_sizes.keys())
        # TensorFlow requires the initial value of a variable to have at least 2-dimensional shape.
        self.ntype_embeddings = {ntype: self.add_weight(name=ntype, shape=(1, node_layer_sizes[ntype]),
                                                        dtype=dtype_tf_float, initializer='random_normal',
                                                        trainable=True) for ntype in contributing_srctypes}
        # self.ntype_embeddings = {ntype: tf.Variable(tf.random.normal((1, node_layer_sizes[ntype]), dtype=dtype_tf_float), name=ntype)
        #                         for ntype in node_layer_sizes if ntype in contributing_srctypes}

    @staticmethod
    def create_layer(edge_layer_sizes, node_layer_sizes):
        edge_layers = HeteroGraphConv.create_layers(edge_layer_sizes)
        node_layers = HeteroGraphConv.create_layers(node_layer_sizes)
        return HeteroGraphConv(edge_layers, node_layers)

    def initial_node_embeddings(self, g):
        return {ntype: tf.tile(e, (g.num_nodes(ntype), 1)) for ntype, e in self.ntype_embeddings.items()}
        #return {ntype: tf.tile(self.ntype_embeddings[ntype], (g.num_nodes(ntype), 1)) for ntype in g.ntypes}

    def call(self, g):
        x = self.initial_node_embeddings(g)
        for layer in self.layers_list:
            x = layer(g, x)
        return {k: x[k] for k in self.output_ntypes}


class HeteroGraphConv(tf.keras.layers.Layer):
    # https://docs.dgl.ai/en/0.4.x/tutorials/basics/5_hetero.html
    # https://docs.dgl.ai/_modules/dgl/nn/tensorflow/conv/relgraphconv.html#RelGraphConv
    # TODO: Add global state. Currently only node states are supported.
    # TODO: Add node features: argument: argument id (or log of the id)

    def __init__(self, edge_layers, node_layers, reduce_func_template=None):
        """
        :param reduce_func_template: A DGL built-in reduce function.
        """
        super().__init__()
        if reduce_func_template is None:
            # Jan Hula suggested that sum is better than mean since it allows the destination node to determine the number of source nodes.
            reduce_func_template = dgl.function.sum
        # Message models
        self.edge_layers = {self.canonical_etype_id(k): v for k, v in edge_layers.items()}
        self.node_layers = node_layers
        self.etype_dict = {}
        for canonical_etype in edge_layers.keys():
            srctype, etype, dsttype = canonical_etype
            msg_func = self.message_func
            reduce_func = reduce_func_template(('m', srctype, etype), ('m', srctype, etype))
            self.etype_dict[canonical_etype] = msg_func, reduce_func
        assert set(tuple(zip(*self.etype_dict.keys()))[2]) == set(self.node_layers.keys())

    def __repr__(self):
        return f'{self.__class__.__name__}({list(self.node_layers.keys())}, {list(self.edge_layers.keys())})'

    @staticmethod
    def create_layers(layer_sizes, create_layer=None):
        if create_layer is None:
            create_layer = functools.partial(tf.keras.layers.Dense, activation='relu')
        return {layer_id: create_layer(units, name=str(layer_id)) for layer_id, units in layer_sizes.items()}

    def message_func(self, edges):
        layer = self.get_edge_layer(edges.canonical_etype)
        input_tensors = [edges.src['h']]
        try:
            input_tensors.append(edges.src['feat'])
        except KeyError:
            pass
        try:
            input_tensors.append(edges.data['feat'])
        except KeyError:
            pass
        srctype, etype, dsttype = edges.canonical_etype
        v = layer(tf.concat(input_tensors, 1))
        return {('m', srctype, etype): v}

    def apply_node_func(self, nodes):
        """Aggregates incoming reduced messages across edge types."""
        input_tensors = [nodes.data['h']]
        for canonical_etype in self.etype_dict.keys():
            srctype, etype, dsttype = canonical_etype
            if dsttype != nodes.ntype:
                continue
            try:
                t = nodes.data[('m', srctype, etype)]
                assert len(t.shape) == 3
                assert t.shape[1] == 1
                t = tf.squeeze(t, axis=1)
            except KeyError:
                edge_layer = self.get_edge_layer(canonical_etype)
                t = tf.zeros((len(nodes), edge_layer.units))
            assert len(t.shape) == 2
            input_tensors.append(t)
        layer = self.node_layers[nodes.ntype]
        v = layer(tf.concat(input_tensors, 1))
        return {'out': v}

    def get_edge_layer(self, canonical_etype):
        return self.edge_layers[self.canonical_etype_id(canonical_etype)]

    @staticmethod
    def canonical_etype_id(canonical_etype):
        return '_'.join(canonical_etype)

    def call(self, g, x):
        """
        Forward computation

        :param g dgl.DGLHeteroGraph: The graph
        :param x dict: Input node features. Maps each node type to a feature tensor.
        :return dict: New node features
        """
        assert set(x.keys()) == set(tuple(zip(*self.etype_dict.keys()))[0])
        with g.local_scope():
            for ntype in x.keys():
                assert ntype in g.ntypes
                assert g.num_nodes(ntype) == len(x[ntype])
                g.nodes[ntype].data['h'] = x[ntype]
            g.multi_update_all(self.etype_dict, 'stack', self.apply_node_func)
            return {ntype: g.nodes[ntype].data['out'] for ntype in self.node_layers.keys()}
