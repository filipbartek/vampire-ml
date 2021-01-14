import collections
import functools
import hashlib
import json

import dgl
import numpy as np
import tensorflow as tf


class FormulaVisitor:
    symbol_types = ('predicate', 'function')

    # Clause roles (a.k.a. input types) according to Unit.hpp:60 `enum InputType`:
    # 0: AXIOM, 1: ASSUMPTION, 2: CONJECTURE, 3: NEGATED_CONJECTURE, 4: CLAIM, 5: EXTENSIONALITY_AXIOM, 6: MODEL_DEFINITION
    # Clause roles according to Unit.cpp:293 `Unit::inputTypeAsString`:
    # 1: hypothesis, 2: negated_conjecture, 5: extensionality, 3: negated_conjecture, default: axiom
    num_clause_roles = 7

    class NumNodesError(RuntimeError):
        def __init__(self, actual, expected):
            super().__init__(f'Invalid number of nodes. Actual: {actual}. Expected: {expected}.')
            self.actual = actual
            self.expected = expected

    def __init__(self, arg_order=True, arg_backedge=True, equality=True, equality_predicate_edge=False,
                 share_terms_global=True, share_terms_local=True, max_number_of_nodes=None, hash_fingerprints=True):
        self.arg_order = arg_order
        self.arg_backedge = arg_backedge
        self.equality = equality
        self.equality_predicate_edge = equality_predicate_edge
        self.terms = None
        if share_terms_global:
            self.terms = {}
        self.share_terms_local = share_terms_local
        self.max_number_of_nodes = max_number_of_nodes
        self.hash_fingerprints = hash_fingerprints
        self.node_counts = collections.defaultdict(lambda: 0)
        # src, dst
        self.edges = {edge_type: ([], []) for edge_type in self.edge_types()}
        self.clause_roles = []
        self.argument_positions = []

    def edge_types(self):
        res = [
            ('formula', 'clause', None),
            ('clause', 'term', 1),
            ('clause', 'term', 0),
            ('clause', 'variable', None),
            ('term', 'predicate', None),
            ('term', 'function', None)
        ]
        if self.arg_order:
            res.extend([
                ('term', 'argument', None),
                ('argument', 'argument', None),
                ('argument', 'term', None),
                ('argument', 'variable', None)
            ])
        else:
            res.extend([
                ('term', 'term', None),
                ('term', 'variable', None)
            ])
        if self.equality:
            res.extend([
                ('clause', 'equality', 1),
                ('clause', 'equality', 0),
                ('equality', 'term', None),
                ('equality', 'variable', None)
            ])
            if self.equality_predicate_edge:
                res.append(('equality', 'predicate', None))
        return res

    def visit_formula(self, formula):
        cur_id_pair = self.add_node('formula')
        for clause in formula:
            clause_id_pair = self.visit_clause(clause)
            self.add_edge(cur_id_pair, clause_id_pair)
        return cur_id_pair

    def visit_clause(self, clause):
        cur_id_pair = self.add_clause(clause['inputType'])
        # To make variables clause-specific, we keep the clause terms separate from the global terms.
        clause_terms = {}
        for literal in clause['literals']:
            atom_id_pair, non_ground = self.visit_term(literal['atom'], clause_terms, cur_id_pair)
            assert atom_id_pair[0] in ('term', 'equality')
            assert literal['polarity'] in (True, False)
            self.add_edge(cur_id_pair, atom_id_pair, literal['polarity'])
        return cur_id_pair

    def visit_term(self, term, clause_terms, clause_id_pair):
        """Generalized term, that is term or atom"""
        fingerprint = json.dumps(term)
        if self.hash_fingerprints:
            fingerprint = hashlib.md5(fingerprint.encode()).hexdigest()
        # This variable signifies that the term is non-ground or that a part of it is shared locally.
        contains_variable = False
        # Term sharing
        if self.terms is not None and fingerprint in self.terms:
            cur_id_pair = self.terms[fingerprint]
        elif fingerprint in clause_terms:
            cur_id_pair = clause_terms[fingerprint]
            contains_variable = True
        else:
            term_type = term['type']
            assert term_type in {'predicate', 'function', 'variable'}
            term_id = term['id']
            assert isinstance(term_id, int)
            if term_type == 'variable':
                assert 'args' not in term
                cur_id_pair = self.add_node('variable')
                contains_variable = True
                # Variables are always shared within a clause.
                clause_terms[fingerprint] = cur_id_pair
                self.add_edge(clause_id_pair, cur_id_pair)
            elif self.equality and term_type == 'predicate' and term_id == 0:
                # Equality
                cur_id_pair = self.add_node('equality')
                if self.equality_predicate_edge:
                    self.add_edge(cur_id_pair, self.add_symbol(term_type, term_id))
                assert len(term['args']) == 2
                for arg in term['args']:
                    arg_id_pair, arg_non_ground = self.visit_term(arg, clause_terms, clause_id_pair)
                    if arg_non_ground:
                        contains_variable = True
                    self.add_edge(cur_id_pair, arg_id_pair)
            else:
                cur_id_pair = self.add_node('term')
                self.add_edge(cur_id_pair, self.add_symbol(term_type, term_id))
                prev_arg_pos_id_pair = None
                for i, arg in enumerate(term['args']):
                    arg_id_pair, arg_non_ground = self.visit_term(arg, clause_terms, clause_id_pair)
                    if arg_non_ground:
                        contains_variable = True
                    if self.arg_order:
                        arg_pos_id_pair = self.add_node('argument')
                        assert arg_pos_id_pair[1] == len(self.argument_positions)
                        self.argument_positions.append(i)
                        self.add_edge(arg_pos_id_pair, arg_id_pair)
                        if prev_arg_pos_id_pair is not None:
                            self.add_edge(prev_arg_pos_id_pair, arg_pos_id_pair)
                        prev_arg_pos_id_pair = arg_pos_id_pair
                        arg_id_pair = arg_pos_id_pair
                    if not self.arg_order or self.arg_backedge or i == 0:
                        self.add_edge(cur_id_pair, arg_id_pair)
            if self.terms is not None and not contains_variable:
                # Across clauses we can only share ground terms.
                self.terms[fingerprint] = cur_id_pair
            elif self.share_terms_local:
                # Within a clause we can share non-ground terms as well as ground terms.
                clause_terms[fingerprint] = cur_id_pair
        return cur_id_pair, contains_variable

    @property
    def number_of_nodes(self):
        return sum(self.node_counts.values())

    def add_clause(self, role):
        assert 0 <= role < self.num_clause_roles
        id_pair = self.add_node('clause')
        assert id_pair[1] == len(self.clause_roles)
        self.clause_roles.append(role)
        return id_pair

    def add_argument(self, position):
        id_pair = self.add_node('argument')
        assert id_pair[1] == len(self.argument_positions)
        self.argument_positions.append(position)
        return id_pair

    def add_symbol(self, symbol_type, symbol_id):
        assert symbol_type in self.symbol_types
        self.node_counts[symbol_type] = max(self.node_counts[symbol_type], symbol_id + 1)
        return symbol_type, symbol_id

    def add_node(self, node_type):
        node_id = self.node_counts[node_type]
        self.node_counts[node_type] += 1
        self.check_number_of_nodes()
        return node_type, node_id

    def check_number_of_nodes(self):
        if self.max_number_of_nodes is not None and self.number_of_nodes > self.max_number_of_nodes:
            raise self.NumNodesError(self.number_of_nodes, self.max_number_of_nodes)

    def add_edge(self, src_id_pair, dst_id_pair, edge_subtype=None):
        src_type, src_id = src_id_pair
        dst_type, dst_id = dst_id_pair
        edge_type_data = self.edges[src_type, dst_type, edge_subtype]
        edge_type_data[0].append(src_id)
        edge_type_data[1].append(dst_id)
        assert len(edge_type_data[0]) == len(edge_type_data[1])

    def graph(self, output_ntypes, node_features=None, feature_argument_positions=False, dtype_id=tf.int32,
              dtype_feat=tf.float32):
        data_dict = {}
        for (src_type, dst_type, edge_subtype), nodes in self.edges.items():
            nodes = tuple(map(functools.partial(self.convert_to_tensor, dtype=dtype_id), nodes))
            for orientation in ('fw', 'bw'):
                if edge_subtype is None:
                    etype = f'{src_type}_{dst_type}_{orientation}'
                else:
                    etype = f'{src_type}_{dst_type}_{orientation}_{edge_subtype}'
                if orientation == 'fw':
                    data_dict[src_type, etype, dst_type] = nodes
                else:
                    assert orientation == 'bw'
                    data_dict[dst_type, etype, src_type] = tuple(reversed(nodes))
        # Add loops on output nodes
        for ntype, num in output_ntypes.items():
            if num is not None:
                assert self.node_counts[ntype] <= num
                self.node_counts[ntype] = max(self.node_counts[ntype], num)
            r = tf.range(self.node_counts[ntype], dtype=dtype_id)
            data_dict[ntype, f'{ntype}_self', ntype] = (r, r)
        g = dgl.heterograph(data_dict, self.node_counts)
        if node_features is not None:
            for ntype, v in node_features.items():
                assert g.num_nodes(ntype) == len(v)
                g.nodes[ntype].data['feat'] = self.convert_to_feature_matrix(v, dtype_feat)
        assert g.num_nodes('clause') == len(self.clause_roles)
        g.nodes['clause'].data['feat'] = tf.one_hot(self.clause_roles, self.num_clause_roles)
        if feature_argument_positions:
            assert g.num_nodes('argument') == len(self.argument_positions)
            g.nodes['argument'].data['feat'] = self.convert_to_feature_matrix(self.argument_positions, dtype_feat)
        return g

    @classmethod
    def convert_to_feature_matrix(cls, v, dtype):
        v = np.asarray(v)
        assert 1 <= len(v.shape) <= 2
        if len(v.shape) == 1:
            v = np.expand_dims(v, 1)
        t = cls.convert_to_tensor(v, dtype=dtype)
        # We avoid reshaping the tensor using TensorFlow.
        # Reshaping the tensor once it's been created could move it to another device.
        assert len(t.shape) == 2
        return t

    @staticmethod
    def convert_to_tensor(v, dtype):
        # Always places the tensor in host memory.
        return tf.convert_to_tensor(v, dtype=dtype)
