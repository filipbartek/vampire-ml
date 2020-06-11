import collections
import functools
import itertools
import json
import logging
import os
import time

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from networkx.algorithms.isomorphism import DiGraphMatcher
from networkx.algorithms.isomorphism import categorical_node_match
from tqdm import tqdm

from proving import utils
from proving.file_path_list import paths_from_patterns
from proving.memory import memory
from proving.solver import Solver
from vampire_ml.results import save_df


def main():
    with joblib.parallel_backend('loky', n_jobs=1):
        # problems = [l.strip('\n') for l in open('data/sp-random-both-clauses/problems.txt').readlines()]
        # problems = list(paths_from_patterns(('PUZ/*-*.p', 'PUZ/*+*.p')))
        problems = list(paths_from_patterns(('**/*-*.p', '**/*+*.p')))
        # problems = ['PUZ/PUZ001-1.p', 'PUZ/PUZ001-2.p', 'PUZ/PUZ001-3.p']
        solver = Solver(timeout=20)
        time_start = time.time()
        graphs = problems_to_graphs(problems, solver)
        logging.debug('Time taken: %s', time.time() - time_start)
        return
        graphs = {k: v for k, v in graphs.items() if v.size() <= 100}
        logging.info('Node counts: %s', [len(g) for g in graphs.values()])
        logging.info('Edge counts: %s', [g.size() for g in graphs.values()])
        apr = Apriori(graphs.values(), minsup=len(graphs) // 2)
        for i, subgraph in enumerate(apr.apriori()):
            out_file = os.path.join('out/dm/subgraphs', f'{i}.svg')
            plot_graph_to_file(subgraph, out_file)


class Apriori:
    def __init__(self, graphs, minsup=1, max_width=None):
        self.graphs = graphs
        self.minsup = minsup
        self.max_width = max_width
        self.rng = np.random.RandomState(0)

    def apriori(self, max_k=None):
        if max_k is None:
            ks = itertools.count()
        else:
            ks = range(max_k)
        for k in ks:
            gs = self.apriori_k(k)
            if len(gs) == 0:
                break
            for g in gs:
                yield g

    @functools.lru_cache(maxsize=2)
    def apriori_k(self, k):
        """Edge-based join growth"""
        assert k >= 0
        logging.info(f'k = {k}:')
        if k == 0:
            res = self.apriori_0()
        elif k == 1:
            res = self.apriori_1()
        else:
            buckets = collections.defaultdict(list)
            for core in tqdm(self.apriori_k(k - 2), unit='core', desc=f'Generating subgraphs for k = {k}'):
                assert core.size() == k - 2
                for g in self.process_core(core, self.apriori_k(k - 1)):
                    assert g.size() == k
                    signature = self.signature(g)
                    if any(is_isomorphic(other, g) for other in buckets[signature]):
                        continue
                    buckets[signature].append(g)
            logging.info(f'Non-empty buckets: {len(buckets)}')
            logging.info(f'Maximum bucket size: {max(len(v) for v in buckets.values())}')
            logging.info(f'Average bucket size: {sum(len(v) for v in buckets.values()) / len(buckets)}')
            # TODO: Remove graphs from `graphs` that support 0 subgraphs.
            res = list(itertools.chain.from_iterable(buckets.values()))
        logging.info(f'Candidate subgraphs: {len(res)}')
        res = self.filter_by_support(res)
        logging.info(f'Frequent subgraphs: {len(res)} / {self.max_width}')
        return res

    @staticmethod
    def signature(g):
        return (len(g),
                tuple(sorted(d for n, d in g.in_degree())),
                tuple(sorted(d for n, d in g.out_degree())))

    @staticmethod
    def process_core(core, prev):
        for i1 in range(len(prev)):
            g1 = prev[i1]
            for iso1 in subgraph_isomorphisms(g1, core):
                assert set(core.nodes) == set(iso1.values())
                assert len(set(g1.nodes) - set(iso1.keys())) <= 1
                assert is_subgraph_isomorphism(g1, core, iso1)
                assert ('new', 0) not in iso1.values()
                g1_common = nx.relabel_nodes(g1, lambda x: iso1.get(x, ('new', 0)))
                assert set(core.edges) <= set(g1_common.edges)
                for i2 in range(i1, len(prev)):
                    g2 = prev[i2]
                    for iso2 in subgraph_isomorphisms(g2, core):
                        assert set(core.nodes) == set(iso2.values())
                        assert len(set(g2.nodes) - set(iso2.keys())) <= 1
                        assert is_subgraph_isomorphism(g2, core, iso2)
                        assert ('new', 1) not in iso2.values()
                        g2_common = nx.relabel_nodes(g2, lambda x: iso2.get(x, ('new', 1)))
                        assert set(core.edges) <= set(g2_common.edges)
                        g = nx.compose(g1_common, g2_common)
                        yield nx.convert_node_labels_to_integers(g)

    def apriori_1(self):
        subgraphs = []
        for nt in itertools.product(node_types, repeat=2):
            g = empty_graph()
            g.add_node(0, type=nt[0])
            g.add_node(1, type=nt[1])
            g.add_edge(0, 1)
            assert all(not is_isomorphic(other, g) for other in subgraphs)
            assert g.size() == 1
            assert len(g) in {1, 2}
            subgraphs.append(g)
        return subgraphs

    def apriori_0(self):
        return [trivial_graph(node_type) for node_type in node_types]

    def filter_by_support(self, subgraphs):
        supports = collections.Counter()
        for g in tqdm(self.graphs, unit='graph', desc='Calculating support'):
            for subgraph in subgraphs:
                if subgraph_is_isomorphic(g, subgraph):
                    supports[subgraph] += 1
        for subgraph, support in list(supports.items()):
            if support < self.minsup:
                del supports[subgraph]
        if len(supports) == 0:
            return []
        return tuple(zip(*supports.most_common(self.max_width)))[0]


def is_subgraph_isomorphism(g1, g2, iso):
    for e1 in itertools.product(iso.keys(), repeat=2):
        assert all(n in g1.nodes for n in e1)
        e2 = (iso[e1[0]], iso[e1[1]])
        if e1 in g1.edges and e2 not in g2.edges:
            logging.debug(f'The edge {e1} is present in g1 but its image {e2} is absent in g2.')
            return False
        if e1 not in g1.edges and e2 in g2.edges:
            logging.debug(f'The edge {e1} is absent in g1 but its image {e2} is present in g2.')
            return False
    return True


def is_isomorphic(g1, g2):
    return graph_matcher(g1, g2).is_isomorphic()


def subgraph_is_isomorphic(big, small):
    """Is a subgraph of big isomorphic to small?"""
    return graph_matcher(big, small).subgraph_is_isomorphic()


def subgraph_isomorphisms(big, small):
    for iso in graph_matcher(big, small).subgraph_isomorphisms_iter():
        assert is_subgraph_isomorphism(big, small, iso)
        yield iso


def graph_matcher(g1, g2):
    return DiGraphMatcher(g1, g2, node_match=categorical_node_match('type', None))


def trivial_graph(node_type):
    g = empty_graph()
    # We use the label "0" to mimic `nx.trivial_graph()`.
    g.add_node(0, type=node_type)
    return g


def empty_graph(name=None):
    return nx.DiGraph(name=name)


def problem_name(path):
    return path.replace('/', '_')


def problems_to_graphs(problems, solver):
    graphs = {}
    records = []
    results = Parallel()(delayed(problem_to_graph)(problem, solver) for problem in
                         tqdm(problems, unit='problem', desc='Generating problem graphs'))
    for problem, (g, record) in zip(problems, results):
        if g is not None:
            graphs[problem] = g
        records.append(record)
    df_problems = utils.dataframe_from_records(records, 'problem')
    save_df(df_problems, 'problems', 'out/dm')
    return graphs


@memory.cache
def problem_to_graph(problem, solver):
    time_start = time.time()
    clausify_result = solver.clausify(problem)
    time_elapsed = time.time() - time_start
    record = {'problem': problem,
              'clausify_returncode': clausify_result.returncode,
              'clausify_time': time_elapsed}
    if clausify_result.returncode != 0:
        return None, record
    symbols = {symbol_type: clausify_result.symbols_of_type(symbol_type) for symbol_type in
               ('predicate', 'function')}
    record.update({'clauses': len(clausify_result.clauses),
                   'predicates': len(symbols['predicate']),
                   'functions': len(symbols['function'])})
    expression_namer = ExpressionNamer(symbols)
    time_start = time.time()
    g = clausify_result_to_graph(clausify_result, expression_namer, name=problem)
    time_elapsed = time.time() - time_start
    record.update({'graph_nodes': len(g), 'graph_edges': g.size(), 'time_graph': time_elapsed})
    if False and len(g) <= 100:
        out_file = os.path.join('out/dm', f'{problem_name(problem)}.svg')
        time_start = time.time()
        plot_graph_to_file(g, out_file)
        record['time_plot'] = time.time() - time_start
    return g, record


def clausify_result_to_graph(clausify_result, expression_namer, name=None):
    g = empty_graph(name=name)
    for symbol_type in ('predicate', 'function'):
        for i in range(len(clausify_result.symbols_of_type(symbol_type))):
            g.add_node(symbol_node_id(symbol_type, i), type=symbol_type, label=expression_namer.symbol(symbol_type, i))
    TermVisitor(g, expression_namer).visit_clauses(clausify_result.clauses)
    return g


def plot_graph_to_file(g, out_file):
    plot_graph(g)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def show_graph(g):
    plot_graph(g, custom_node_labels=False)
    plt.show()
    plt.close()


def plot_graph(g, custom_node_labels=True):
    plt.figure(figsize=(12, 8))
    plt.title(g.name)
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    node_colors = [node_color(node[1]) for node in g.nodes.data('type')]
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, cmap='Pastel1', vmin=min(node_color_dict.values()),
                           vmax=max(node_color_dict.values()))
    node_labels = None
    if custom_node_labels:
        node_labels = {k: node_label(v) for k, v in g.nodes.data()}
    nx.draw_networkx_labels(g, pos, font_size=8, labels=node_labels)
    edge_colors = [edge_color(e[2]) for e in g.edges(data=True)]
    nx.draw_networkx_edges(g, pos, edge_color=edge_colors)


"""
Possible edges:
- root -> clause
- clause -> atom [polarity=0]
- clause -> atom [polarity=1]
- atom -> equality
- atom -> predicate
- With argument ordering:
    - atom -> argument
    - argument -> argument
    - argument -> term
    - argument -> variable
- term -> function
"""
node_types = [
    'root',
    'clause',
    'argument',
    'atom',
    'equality',
    'predicate',
    'term',
    'function',
    'variable'
]

node_color_dict = {node_type: i for i, node_type in enumerate(node_types)}


def node_color(node_type):
    try:
        return node_color_dict[node_type]
    except KeyError:
        return None


def node_label(attributes):
    try:
        return attributes['label']
    except KeyError:
        return ''


def edge_color(attributes):
    if 'type' in attributes and attributes['type'] == 'clause_atom':
        if attributes['polarity']:
            return 'red'
        else:
            return 'blue'
    return 'black'


class TermVisitor:
    def __init__(self, graph, expression_namer, arg_order=True, arg_backedge=True, root_node=False):
        self.graph = graph
        self.expression_namer = expression_namer
        self.arg_order = arg_order
        self.arg_backedge = arg_backedge
        self.root_node = root_node

    def visit_clauses(self, clauses):
        if self.root_node:
            root_id = None
            assert root_id not in self.graph
            self.graph.add_node(root_id, type='root')
        for i, clause in enumerate(clauses):
            clause_id = self.visit_clause(clause, i)
            assert clause_id in self.graph
            if self.root_node:
                self.graph.add_edge(root_id, clause_id, type='root_clause')

    def visit_clause(self, clause, i):
        root_id = ('clause', i)
        if root_id not in self.graph:
            self.graph.add_node(root_id, type='clause', input_type=clause['inputType'], label=i)
            for literal in clause['literals']:
                atom_id = self.visit_term(literal['atom'])
                assert atom_id in self.graph
                self.graph.add_edge(root_id, atom_id, type='clause_atom', polarity=literal['polarity'])
        return root_id

    def visit_term(self, term):
        """Generalized term, that is term or atom"""
        root_id = ('term', json.dumps(term))
        if root_id not in self.graph:
            term_type = term['type']
            assert term_type in {'predicate', 'function', 'variable'}
            if term_type == 'variable':
                assert 'args' not in term
                self.graph.add_node(root_id, type='variable', label=self.expression_namer.variable(term['id']))
            elif term_type == 'predicate' and term['id'] == 0:
                # Equality
                self.graph.add_node(root_id, type='equality', label='=')
                symbol_id = symbol_node_id(term_type, term['id'])
                assert symbol_id in self.graph
                self.graph.add_edge(root_id, symbol_id, type='application_symbol')
                assert len(term['args']) == 2
                for arg in term['args']:
                    arg_id = self.visit_term(arg)
                    assert arg_id in self.graph
                    self.graph.add_edge(root_id, arg_id, type='equality_argument')
            else:
                node_type = {'predicate': 'atom', 'function': 'term'}[term_type]
                self.graph.add_node(root_id, type=node_type, label=self.expression_namer.term(term))
                symbol_id = symbol_node_id(term_type, term['id'])
                assert symbol_id in self.graph
                self.graph.add_edge(root_id, symbol_id, type='application_symbol')
                prev_arg_pos_id = None
                for i, arg in enumerate(term['args']):
                    arg_id = self.visit_term(arg)
                    if self.arg_order:
                        # Insert argument ordering node.
                        arg_pos_id = ('argument', root_id[1], i)
                        self.graph.add_node(arg_pos_id, type='argument', label=i)
                        self.graph.add_edge(arg_pos_id, arg_id, type='argument_term')
                        if prev_arg_pos_id is not None:
                            self.graph.add_edge(prev_arg_pos_id, arg_pos_id, type='argument_argument')
                        prev_arg_pos_id = arg_pos_id
                        arg_id = arg_pos_id
                    if not self.arg_order or self.arg_backedge or i == 0:
                        assert arg_id in self.graph
                        self.graph.add_edge(root_id, arg_id, type='application_argument')
        assert root_id in self.graph
        return root_id


def symbol_node_id(symbol_type, symbol_id):
    return symbol_type, symbol_id


class ExpressionNamer:
    def __init__(self, symbols):
        self.symbols = symbols

    def clause(self, clause):
        return '|'.join(self.literal(literal) for literal in clause['literals'])

    def literal(self, literal):
        atom_str = self.term(literal['atom'])
        if literal['polarity']:
            return atom_str
        else:
            return f'~{atom_str}'

    def term(self, term):
        """Generalized term, that is term or atom"""
        term_type = term['type']
        assert term_type in {'predicate', 'function', 'variable'}
        term_id = term['id']
        if term_type == 'variable':
            return f'x{term_id}'
        name = self.symbol(term_type, term_id)
        args_str = ','.join(self.term(arg) for arg in term['args'])
        return f'{name}({args_str})'

    def symbol(self, symbol_type, symbol_id):
        return self.symbols[symbol_type]['name'][symbol_id]

    @staticmethod
    def variable(variable_id):
        return f'x{variable_id}'


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(threadName)s %(levelname)s - %(message)s')
    # logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    main()
