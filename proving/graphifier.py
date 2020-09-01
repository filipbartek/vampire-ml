import collections
import itertools
import json
import logging
import time

import dgl
import tensorflow as tf
from joblib import Parallel, delayed

from proving import tptp
from proving.memory import memory
from proving.utils import number_of_nodes


# For example HWV091_1 takes 21.1s to graphify.
@memory.cache
def problem_to_graph(graphifier, problem):
    return graphifier.problem_to_graph(problem)


class Graphifier:
    """Stateless. Can be reused to create multiple graphs."""
    edge_type_forward = 'forward'
    edge_type_backward = 'backward'
    edge_type_self = 'self'

    def __init__(self, solver, arg_order=True, arg_backedge=True, equality=True, max_number_of_nodes=None):
        self.solver = solver
        self.arg_order = arg_order
        self.arg_backedge = arg_backedge
        self.equality = equality
        self.max_number_of_nodes = max_number_of_nodes

        self.ntypes = ['clause', 'term', 'predicate', 'function', 'variable']
        ntype_pairs = [
            ('clause', 'term'),
            ('term', 'predicate'),
            ('term', 'function')
        ]
        if arg_order:
            self.ntypes.append('argument')
            ntype_pairs.extend([
                ('term', 'argument'),
                ('argument', 'argument'),
                ('argument', 'term'),
                ('argument', 'variable')
            ])
        else:
            ntype_pairs.extend([
                ('term', 'term'),
                ('term', 'variable')
            ])
        if equality:
            self.ntypes.append('equality')
            ntype_pairs.extend([
                ('clause', 'equality'),
                ('equality', 'term'),
                ('equality', 'variable')
            ])
        assert all(ntype in self.ntypes for ntype in itertools.chain.from_iterable(ntype_pairs))

        self.canonical_etypes = []
        for srctype, dsttype in ntype_pairs:
            self.canonical_etypes.extend(
                ((srctype, self.edge_type_forward, dsttype), (dsttype, self.edge_type_backward, srctype)))
        # Self-loops
        self.canonical_etypes.extend(
            (('predicate', self.edge_type_self, 'predicate'), ('function', self.edge_type_self, 'function')))

    def __repr__(self):
        attrs = ('solver', 'arg_order', 'arg_backedge', 'equality', 'max_number_of_nodes')
        return f'{self.__class__.__name__}(%s)' % ', '.join(f'{k}={getattr(self, k)}' for k in attrs)

    def problems_to_graphs(self, problems):
        logging.info(f'Graphifying {len(problems)} problems...')
        res = Parallel(verbose=1)(delayed(problem_to_graph)(self, problem) for problem in problems)
        logging.info(
            f'Problems graphified. {sum(g is not None for g, record in res)}/{len(res)} graphified successfully.')
        return res

    def problem_to_graph(self, problem):
        time_start = time.time()
        clausify_result = self.solver.clausify(problem)
        time_elapsed = time.time() - time_start
        record = {'problem': problem,
                  'clausify_returncode': clausify_result.returncode,
                  'clausify_time': time_elapsed}
        record.update(tptp.problem_properties(problem))
        if clausify_result.returncode != 0 or clausify_result.clauses is None or clausify_result.symbols is None:
            return None, record
        symbol_types = ('predicate', 'function')
        symbols = {symbol_type: clausify_result.symbols_of_type(symbol_type) for symbol_type in symbol_types}
        record['clause_count'] = len(clausify_result.clauses)
        record.update({f'{symbol_type}_count': len(symbols[symbol_type]) for symbol_type in symbol_types})
        time_start = time.time()
        try:
            g = self.clausify_result_to_graph(clausify_result)
        except RuntimeError:
            # The graph would be too large (too many nodes).
            g = None
        time_elapsed = time.time() - time_start
        record['graph_time'] = time_elapsed
        if g is not None:
            record['graph_nodes'] = number_of_nodes(g)
            record.update({f'graph_nodes_{ntype}': g.number_of_nodes(ntype) for ntype in g.ntypes})
            record['graph_edges'] = sum(g.number_of_edges(canonical_etype) for canonical_etype in g.canonical_etypes)
        return g, record

    def clausify_result_to_graph(self, clausify_result):
        symbol_features = {symbol_type: clausify_result.symbols_of_type(symbol_type)[['inGoal', 'introduced']] for
                           symbol_type in ('predicate', 'function')}
        tv = TermVisitor(self, symbol_features)
        tv.visit_clauses(clausify_result.clauses)
        return tv.get_graph()


class TermVisitor:
    """Stateful. Must be instantiated for each new graph."""

    def __init__(self, template, symbol_features):
        # TODO: Allow limiting term sharing to inside a clause, letting every clause to use a separate set of variables.
        # TODO: Add a root "formula" node that connects to all clauses.
        # TODO: Allow introducing separate node types for predicates and functions.
        # TODO: Allow disabling term sharing.
        # TODO: Add global state (node) for equation.
        # TODO: Encode edge polarity as edge feature.
        # TODO: Encode symbol node features: isFunction, inGoal, introduced
        # TODO: Maybe also: arity, usageCnt, unitUsageCnt, inUnit
        # TODO: Clause feature: inGoal
        # TODO: Node feature: argument: argument id (or log of the id)
        self.template = template
        self.node_counts = {node_type: 0 for node_type in self.template.ntypes}
        self.edges = {edge_type: [] for edge_type in self.template.canonical_etypes}
        self.terms = {}
        self.node_features = collections.defaultdict(list)
        self.edge_features = collections.defaultdict(list)
        for symbol_type, feat in symbol_features.items():
            # We add self-loops for all symbol nodes so that even nodes without any connection to the graph are preserved.
            self.edges[symbol_type, self.edge_type_self, symbol_type] = [(i, i) for i in range(len(feat))]
            self.node_counts[symbol_type] = len(feat)
            self.node_features[symbol_type] = feat
        self.check_number_of_nodes()

    @property
    def number_of_nodes(self):
        return sum(self.node_counts.values())

    def check_number_of_nodes(self):
        if self.max_number_of_nodes is not None and self.number_of_nodes > self.max_number_of_nodes:
            raise RuntimeError('Too many nodes.')

    @property
    def edge_type_forward(self):
        return self.template.edge_type_forward

    @property
    def edge_type_backward(self):
        return self.template.edge_type_backward

    @property
    def edge_type_self(self):
        return self.template.edge_type_self

    @property
    def arg_order(self):
        return self.template.arg_order

    @property
    def arg_backedge(self):
        return self.template.arg_backedge

    @property
    def equality(self):
        return self.template.equality

    @property
    def max_number_of_nodes(self):
        return self.template.max_number_of_nodes

    def get_data_dict(self):
        assert set(self.edges.keys()) == set(self.template.canonical_etypes)
        return self.edges

    def get_graph(self):
        g = dgl.heterograph(self.get_data_dict(), self.node_counts)
        for ntype, v in self.node_features.items():
            assert g.number_of_nodes(ntype) == len(v)
            g.nodes[ntype].data['feat'] = self.convert_to_matrix(v)
        for (srctype, dsttype), v in self.edge_features.items():
            canonical_etype = srctype, self.edge_type_forward, dsttype
            assert g.number_of_edges(canonical_etype) == len(v)
            g.edges[canonical_etype].data['feat'] = self.convert_to_matrix(v)
            canonical_etype = dsttype, self.edge_type_backward, srctype
            assert g.number_of_edges(canonical_etype) == len(v)
            g.edges[canonical_etype].data['feat'] = self.convert_to_matrix(v)
        return g

    @staticmethod
    def convert_to_matrix(v):
        t = tf.convert_to_tensor(v, dtype=tf.keras.backend.floatx())
        assert 1 <= len(t.shape) <= 2
        if len(t.shape) == 1:
            t = tf.expand_dims(t, 1)
        return t

    def visit_clauses(self, clauses):
        for clause in clauses:
            self.visit_clause(clause)

    def visit_clause(self, clause):
        cur_id_pair = self.add_node('clause')
        for literal in clause['literals']:
            atom_id_pair = self.visit_term(literal['atom'])
            assert atom_id_pair[0] in ('term', 'equality')
            assert literal['polarity'] in (True, False)
            self.add_edge(cur_id_pair, atom_id_pair, literal['polarity'])

    def visit_term(self, term):
        """Generalized term, that is term or atom"""
        root_id = json.dumps(term)
        if root_id in self.terms:
            # Term sharing
            cur_id_pair = self.terms[root_id]
        else:
            term_type = term['type']
            assert term_type in {'predicate', 'function', 'variable'}
            term_id = term['id']
            assert isinstance(term_id, int)
            if term_type == 'variable':
                assert 'args' not in term
                # TODO: Expose the mapping from variable ids (`term_id`) to node ids.
                cur_id_pair = self.add_node('variable')
            elif self.equality and term_type == 'predicate' and term_id == 0:
                # Equality
                cur_id_pair = self.add_node('equality')
                assert len(term['args']) == 2
                for arg in term['args']:
                    self.add_edge(cur_id_pair, self.visit_term(arg))
            else:
                cur_id_pair = self.add_node('term')
                symbol_id_pair = self.add_symbol(term_type, term_id)
                self.add_edge(cur_id_pair, symbol_id_pair)
                prev_arg_pos_id_pair = None
                for i, arg in enumerate(term['args']):
                    arg_id_pair = self.visit_term(arg)
                    if self.arg_order:
                        arg_pos_id_pair = self.add_node('argument')
                        self.add_edge(arg_pos_id_pair, arg_id_pair)
                        if prev_arg_pos_id_pair is not None:
                            self.add_edge(prev_arg_pos_id_pair, arg_pos_id_pair)
                        prev_arg_pos_id_pair = arg_pos_id_pair
                        arg_id_pair = arg_pos_id_pair
                    if not self.arg_order or self.arg_backedge or i == 0:
                        self.add_edge(cur_id_pair, arg_id_pair)
            self.terms[root_id] = cur_id_pair
        assert root_id in self.terms and self.terms[root_id] == cur_id_pair
        return cur_id_pair

    def add_node(self, node_type):
        node_id = self.node_counts[node_type]
        self.node_counts[node_type] += 1
        self.check_number_of_nodes()
        return node_type, node_id

    def add_symbol(self, symbol_type, symbol_id):
        assert symbol_type in ('predicate', 'function')
        assert (symbol_id, symbol_id) in self.edges[symbol_type, self.edge_type_self, symbol_type]
        assert 0 <= symbol_id < len(self.node_features[symbol_type])
        return symbol_type, symbol_id

    def add_edge(self, src_id_pair, dst_id_pair, features=None):
        src_type, src_id = src_id_pair
        dst_type, dst_id = dst_id_pair
        self.edges[src_type, self.edge_type_forward, dst_type].append((src_id, dst_id))
        self.edges[dst_type, self.edge_type_backward, src_type].append((dst_id, src_id))
        if features is not None:
            self.edge_features[src_type, dst_type].append(features)
