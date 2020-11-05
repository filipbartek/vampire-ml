import functools
import itertools
import json
import logging
import time

import dgl
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

from proving import tptp
from proving.memory import memory


# For example HWV091_1 takes 21.1s to graphify.
@memory.cache
def problem_to_graph(graphifier, problem):
    return graphifier.problem_to_graph(problem)


class Graphifier:
    """Stateless. Can be reused to create multiple graphs."""
    edge_type_forward = 'forward'
    edge_type_backward = 'backward'
    edge_type_self = 'self'

    def __init__(self, solver, arg_order=True, arg_backedge=True, equality=True, max_number_of_nodes=None,
                 output_ntypes=('predicate', 'function')):
        self.solver = solver
        self.arg_order = arg_order
        self.arg_backedge = arg_backedge
        self.equality = equality
        self.max_number_of_nodes = max_number_of_nodes
        self.output_ntypes = output_ntypes

        self.ntypes = ['clause', 'term', 'predicate', 'function', 'variable']
        self.ntype_pairs = [
            ('clause', 'term'),
            ('term', 'predicate'),
            ('term', 'function')
        ]
        if arg_order:
            self.ntypes.append('argument')
            self.ntype_pairs.extend([
                ('term', 'argument'),
                ('argument', 'argument'),
                ('argument', 'term'),
                ('argument', 'variable')
            ])
        else:
            self.ntype_pairs.extend([
                ('term', 'term'),
                ('term', 'variable')
            ])
        if equality:
            self.ntypes.append('equality')
            self.ntype_pairs.extend([
                ('clause', 'equality'),
                ('equality', 'term'),
                ('equality', 'variable')
            ])
        assert set(self.output_ntypes) <= set(self.ntypes)
        assert set(itertools.chain.from_iterable(self.ntype_pairs)) <= set(self.ntypes)

    def __getitem__(self, problem):
        return problem_to_graph(self, problem)[0]

    @property
    def canonical_etypes(self):
        res_standard = itertools.chain.from_iterable(
            ((srctype, self.edge_type_forward, dsttype), (dsttype, self.edge_type_backward, srctype)) for
            srctype, dsttype in self.ntype_pairs)
        # We add loops to potential output ntypes to ensure that their embeddings are always computed.
        res_self = ((ntype, self.edge_type_self, ntype) for ntype in self.output_ntypes)
        return list(itertools.chain(res_standard, res_self))

    def __repr__(self):
        attrs = ('solver', 'arg_order', 'arg_backedge', 'equality', 'max_number_of_nodes')
        return f'{self.__class__.__name__}(%s)' % ', '.join(f'{k}={getattr(self, k)}' for k in attrs)

    def problems_to_batch_graph(self, problems):
        if isinstance(problems, tf.Tensor):
            problems = tuple(bytes.decode(p.numpy()) for p in problems)
        return self._problems_to_batch_graph(problems)

    @functools.lru_cache(maxsize=None)
    def _problems_to_batch_graph(self, problems):
        graphs = self.problems_to_graphs(problems, return_records=False)
        batch_graph = dgl.batch(graphs)
        return batch_graph

    def problems_to_graphs(self, problems, return_records=True):
        logging.debug(f'Graphifying {len(problems)} problems...')
        res = Parallel(verbose=0)(delayed(problem_to_graph)(self, problem) for problem in problems)
        logging.debug(
            f'Problems graphified. {sum(g is not None for g, record in res)}/{len(res)} graphified successfully.')
        if not return_records:
            res = [r[0] for r in res]
        return res

    def problem_to_graph(self, problem):
        if isinstance(problem, tf.Tensor):
            problem = bytes.decode(problem.numpy())
        logging.debug(f'Graphifying problem {problem}...')
        time_start = time.time()
        clausify_result = self.solver.clausify(problem)
        time_elapsed = time.time() - time_start
        record = {'problem': problem,
                  'clausify_returncode': clausify_result.returncode,
                  'clausify_time': time_elapsed}
        record.update(tptp.problem_properties(problem))
        if clausify_result.returncode != 0 or clausify_result.clauses is None or clausify_result.symbols is None:
            logging.debug(f'Failed to graphify problem {problem}: clausification failed.')
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
            logging.debug(f'Failed to graphify problem {problem}.', exc_info=True)
            g = None
        time_elapsed = time.time() - time_start
        record['graph_time'] = time_elapsed
        if g is not None:
            record['graph_nodes'] = g.num_nodes()
            record.update({f'graph_nodes_{ntype}': g.num_nodes(ntype) for ntype in g.ntypes})
            record['graph_edges'] = sum(g.num_edges(canonical_etype) for canonical_etype in g.canonical_etypes)
            logging.debug(f'Problem {problem} graphified.')
        return g, record

    def clausify_result_to_graph(self, clausify_result):
        symbol_features = {symbol_type: clausify_result.symbols_of_type(symbol_type)[['inGoal', 'introduced']] for
                           symbol_type in ('predicate', 'function')}
        tv = TermVisitor(self, symbol_features)
        tv.visit_clauses(clausify_result.clauses)
        return tv.get_graph()


class TermVisitor:
    """Stateful. Must be instantiated for each new graph."""

    def __init__(self, template, node_features=None, feature_dtype=tf.float32):
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
        self.feature_dtype = feature_dtype
        self.node_counts = {node_type: 0 for node_type in self.template.ntypes}
        if node_features is None:
            node_features = {}
        for ntype, feat in node_features.items():
            self.node_counts[ntype] = len(feat)
        self.edges = {ntype_pair: {'src': [], 'dst': []} for ntype_pair in self.template.ntype_pairs}
        for ntype_pair in (('clause', 'term'), ('clause', 'equality')):
            if ntype_pair in self.edges:
                self.edges[ntype_pair]['feat'] = []
        self.terms = {}
        self.node_features = node_features
        self.check_number_of_nodes()

    @property
    def number_of_nodes(self):
        return sum(self.node_counts.values())

    def check_number_of_nodes(self):
        if self.max_number_of_nodes is not None and self.number_of_nodes > self.max_number_of_nodes:
            raise RuntimeError(f'Too many nodes. Current: {self.number_of_nodes}. Maximum: {self.max_number_of_nodes}.')

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

    def get_graph(self):
        dtype = tf.int32
        data_dict = {}
        edge_features = {}
        for (srctype, dsttype), d in self.edges.items():
            canonical_etype_fw = srctype, self.edge_type_forward, dsttype
            canonical_etype_bw = dsttype, self.edge_type_backward, srctype
            src = self.convert_to_tensor(d['src'], dtype=dtype)
            dst = self.convert_to_tensor(d['dst'], dtype=dtype)
            data_dict[canonical_etype_fw] = (src, dst)
            data_dict[canonical_etype_bw] = (dst, src)
            if 'feat' in d:
                feat = self.convert_to_feature_matrix(d['feat'])
                edge_features[canonical_etype_fw] = feat
                edge_features[canonical_etype_bw] = feat
        # Loops
        for ntype in self.template.output_ntypes:
            r = tf.range(self.node_counts[ntype], dtype=dtype)
            data_dict[ntype, self.edge_type_self, ntype] = (r, r)
        g = dgl.heterograph(data_dict, self.node_counts)
        for ntype, v in self.node_features.items():
            assert g.num_nodes(ntype) == len(v)
            g.nodes[ntype].data['feat'] = self.convert_to_feature_matrix(v)
        for canonical_etype, v in edge_features.items():
            assert g.num_edges(canonical_etype) == len(v)
            g.edges[canonical_etype].data['feat'] = v
        return g

    def convert_to_feature_matrix(self, v):
        v = np.asarray(v)
        assert 1 <= len(v.shape) <= 2
        if len(v.shape) == 1:
            v = np.expand_dims(v, 1)
        t = self.convert_to_tensor(v, dtype=self.feature_dtype)
        # We avoid reshaping the tensor using TensorFlow.
        # Reshaping the tensor once it's been created could move it to another device.
        assert len(t.shape) == 2
        return t

    @staticmethod
    def convert_to_tensor(v, dtype):
        # Always places the tensor in host memory.
        return tf.convert_to_tensor(v, dtype=dtype)

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
        assert 0 <= symbol_id < len(self.node_features[symbol_type])
        return symbol_type, symbol_id

    def add_edge(self, src_id_pair, dst_id_pair, features=None):
        src_type, src_id = src_id_pair
        dst_type, dst_id = dst_id_pair
        edge_type_data = self.edges[src_type, dst_type]
        edge_type_data['src'].append(src_id)
        edge_type_data['dst'].append(dst_id)
        assert ('feat' in edge_type_data) == (features is not None)
        if features is not None:
            edge_type_data['feat'].append(features)
            assert len(edge_type_data['src']) == len(edge_type_data['dst']) == len(edge_type_data['feat'])
        assert len(edge_type_data['src']) == len(edge_type_data['dst'])
