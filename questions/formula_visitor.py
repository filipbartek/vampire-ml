import collections
import functools
import hashlib
import itertools
import json

import dgl
import numpy as np
import tensorflow as tf
from ordered_set import OrderedSet


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

    def __init__(self, arg_order=True, arg_backedge=True, formula_nodes=True, atom_nodes=True,
                 equality=True, equality_predicate_edge=False,
                 share_terms_global=True, share_terms_local=True, max_number_of_nodes=None, hash_fingerprints=True):
        self.arg_order = arg_order
        self.arg_backedge = arg_backedge
        self.formula_nodes = formula_nodes
        if atom_nodes:
            self.ntype_atom = 'atom'
        else:
            self.ntype_atom = 'term'
        self.equality_nodes = equality
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

    def edge_types(self):
        res = OrderedSet()
        if self.formula_nodes:
            res.add(('formula', 'clause', None))
        res.add(('clause', 'variable', None))
        res.update([
            ('clause', self.ntype_atom, 1),
            ('clause', self.ntype_atom, 0),
            (self.ntype_atom, 'predicate', None)
        ])
        res.add(('term', 'function', None))
        if self.arg_order:
            res.update([
                (self.ntype_atom, 'argument', None),
                ('term', 'argument', None),
                ('argument', 'argument', None),
                ('argument', 'term', None),
                ('argument', 'variable', None)
            ])
        else:
            res.update([
                (self.ntype_atom, 'term', None),
                (self.ntype_atom, 'variable', None),
                ('term', 'term', None),
                ('term', 'variable', None)
            ])
        if self.equality_nodes:
            res.update([
                ('clause', 'equality', 1),
                ('clause', 'equality', 0),
                ('equality', 'term', None),
                ('equality', 'variable', None)
            ])
            if self.equality_predicate_edge:
                res.add(('equality', 'predicate', None))
        return res

    def ntypes(self):
        return OrderedSet(itertools.chain.from_iterable((e[i] for e in self.edge_types()) for i in (0, 1)))

    def visit_formula(self, formula):
        cur_id_pair = None
        if self.formula_nodes:
            cur_id_pair = self.add_node('formula')
        for clause in formula:
            clause_id_pair = self.visit_clause(clause)
            if cur_id_pair is not None:
                self.add_edge(cur_id_pair, clause_id_pair)
        return cur_id_pair

    def visit_clause(self, clause):
        cur_id_pair = self.add_clause(clause['inputType'])
        # To make variables clause-specific, we keep the clause terms separate from the global terms.
        clause_terms = {}
        for literal in clause['literals']:
            atom_id_pair, non_ground = self.visit_term(literal['atom'], clause_terms, cur_id_pair)
            assert atom_id_pair[0] in (self.ntype_atom, 'equality')
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
            elif self.equality_nodes and term_type == 'predicate' and term_id == 0:
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
                ntype_term = 'term'
                if term_type == 'predicate':
                    ntype_term = self.ntype_atom
                cur_id_pair = self.add_node(ntype_term)
                self.add_edge(cur_id_pair, self.add_symbol(term_type, term_id))
                prev_arg_pos_id_pair = None
                for i, arg in enumerate(term['args']):
                    arg_id_pair, arg_non_ground = self.visit_term(arg, clause_terms, clause_id_pair)
                    if arg_non_ground:
                        contains_variable = True
                    if self.arg_order:
                        arg_pos_id_pair = self.add_node('argument')
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

    etype_ntype_pairs = {
        'formula_contains_clause': [('formula', 'clause')],
        'clause_contains_atom': [('clause', 'atom'), ('clause', 'equality')],
        'clause_binds_variable': [('clause', 'variable')],
        'atom_applies_predicate': [('atom', 'predicate'), ('equality', 'predicate')],
        'term_applies_function': [('term', 'function')],
        'atom_applies_on_argument': [('atom', 'argument')],
        'atom_applies_on_term': [('atom', 'term'), ('atom', 'variable')],
        'term_applies_on_argument': [('term', 'argument')],
        'term_applies_on_term': [('term', 'term'), ('term', 'variable')],
        'equality_equalizes_argument': [('equality', 'argument')],
        'equality_equalizes_term': [('equality', 'term'), ('equality', 'variable')],
        'argument_is_term': [('argument', 'term'), ('argument', 'variable')],
        'argument_precedes_argument': [('argument', 'argument')]
    }

    etype_norms = {
        # formula->clause: no normalization because each clause is adjacent to only one formula
        'formula_contains_clause': ('none', 'right'),
        # clause->atom: no normalization so that the atom knows its number of occurrences (term sharing)
        # atom->clause: no normalization because both atom and equality share this etype, and there are polarity-based subtypes
        'clause_contains_atom': ('none', 'none'),
        'clause_binds_variable': ('none', 'none'),
        # atom->predicate: we don't normalize so that the predicate knows its number of occurrences
        'atom_applies_predicate': ('none', 'none'),
        'term_applies_function': ('none', 'none'),
        'atom_applies_on_argument': ('none', 'right'),
        'atom_applies_on_term': ('both', 'both'),
        'term_applies_on_argument': ('none', 'right'),
        'term_applies_on_term': ('both', 'both'),
        'equality_equalizes_argument': ('none', 'none'),
        'equality_equalizes_term': ('right', 'none'),
        # argument->term: no normalization so that the term knows how many times it occurs (term sharing)
        'argument_is_term': ('none', 'none'),
        'argument_precedes_argument': ('none', 'none')
    }

    # https://arxiv.org/pdf/1703.06103.pdf : 'inv'
    inverse_token = 'inverse'

    @classmethod
    def conv_norm(cls, etype):
        orientation = 0
        split = etype.rsplit(sep='_', maxsplit=1)
        if split[1] == 'self':
            return 'none'
        if split[1] == cls.inverse_token:
            etype = split[0]
            orientation = 1
        split = etype.rsplit(sep='_', maxsplit=1)
        if split[0] == 'clause_contains_atom':
            etype = split[0]
        return cls.etype_norms[etype][orientation]

    @classmethod
    @functools.lru_cache(maxsize=1)
    def ntype_pair_to_etype(cls):
        res = {}
        for etype, ntype_pairs in cls.etype_ntype_pairs.items():
            for ntype_pair in ntype_pairs:
                assert ntype_pair not in res
                res[ntype_pair] = etype
        return res

    @classmethod
    def etype(cls, stype, dtype, orientation, edge_subtype=None):
        etype = cls.ntype_pair_to_etype()[stype, dtype]
        if edge_subtype is not None:
            etype = f'{etype}_{edge_subtype}'
        if orientation <= 0:
            etype = f'{etype}_{cls.inverse_token}'
        return etype

    def graph(self, output_ntypes, node_features=None, dtype_id=tf.int32, dtype_feat=tf.float32, device='/cpu'):
        """
        :param device: TensorFlow device to store the graph on.
        The default device is '/cpu' because the graphs are disk-cached and when a graph is deserialized from disk,
        it is always stored on '/cpu'.
        """
        with tf.device(device):
            data_dict = {}
            for (src_type, dst_type, edge_subtype), nodes in self.edges.items():
                nodes = tuple(map(functools.partial(self.convert_to_tensor, dtype=dtype_id), nodes))
                for orientation in (1, -1):
                    etype = self.etype(src_type, dst_type, orientation, edge_subtype)
                    if orientation == 1:
                        data_dict[src_type, etype, dst_type] = nodes
                    else:
                        assert orientation == -1
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
        result = tf.convert_to_tensor(v, dtype=dtype)
        # Moves the data to an appropriate device. See https://github.com/tensorflow/tensorflow/issues/42242
        result = tf.identity(result)
        return result
