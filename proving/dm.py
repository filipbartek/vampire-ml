#!/usr/bin/env python3

import argparse
import collections
import functools
import itertools
import json
import logging
import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sklearn.metrics
from joblib import Parallel, delayed
from networkx.algorithms.isomorphism import DiGraphMatcher
from networkx.algorithms.isomorphism import categorical_node_match, categorical_edge_match
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.manifold import Isomap
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

from proving import file_path_list
from proving import tptp
from proving import utils
from proving.memory import memory
from proving.solver import Solver
from vampire_ml.results import save_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='out/dm')
    parser.add_argument('--problem', action='append', default=[])
    parser.add_argument('--problem-list', action='append', default=[])
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--max-problems', type=int)
    parser.add_argument('--no-mining', action='store_true')
    parser.add_argument('--no-featurization', action='store_true')
    parser.add_argument('--mining-max-graphs', type=int)
    parser.add_argument('--mining-max-graph-size', type=int)
    parser.add_argument('--mining-min-subgraph-size', type=int, default=0)
    parser.add_argument('--mining-max-subgraph-size', type=int)
    parser.add_argument('--mining-min-support', type=float, default=0.25)
    parser.add_argument('--mining-max-support', type=float, default=1.00)
    parser.add_argument('--mining-max-subgraphs-per-level', type=int)
    parser.add_argument('--mining-only-maximal-subgraphs', action='store_true')
    parser.add_argument('--featurization-max-graph-size', type=int)
    parser.add_argument('--plot-mining-graphs', action='store_true')
    parser.add_argument('--plot-frequent-subgraphs', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(threadName)s %(levelname)s - %(message)s')
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    dtypes = col_dtypes.copy()

    with joblib.parallel_backend('threading', n_jobs=args.jobs):
        problems, _ = list(file_path_list.compose(args.problem_list, args.problem))
        problems = np.asarray(problems)
        logging.info('Problems available: %s', len(problems))
        if args.max_problems is not None and len(problems) > args.max_problems:
            problems = np.random.RandomState(0).choice(problems, size=args.max_problems, replace=False)
        logging.info('Problems under consideration: %s', len(problems))
        solver = Solver(timeout=20)
        assert args.featurization_max_graph_size is None or args.mining_max_graph_size is None or args.featurization_max_graph_size >= args.mining_max_graph_size
        graphs, records = zip(*problems_to_graphs(problems, solver, max_size=args.featurization_max_graph_size))
        assert len(problems) == len(graphs) == len(records)
        all_graph_indices = [i for i, g in enumerate(graphs) if g is not None]
        logging.info('Graphs total: %s', len(all_graph_indices))
        all_graph_indices = sorted(all_graph_indices, key=lambda i: graphs[i].size(), reverse=True)
        mining_indices = all_graph_indices
        if args.mining_max_graph_size is not None:
            mining_indices = [i for i in mining_indices if graphs[i].size() <= args.mining_max_graph_size]
        if args.mining_max_graphs is not None and len(mining_indices) > args.mining_max_graphs:
            mining_indices = np.asarray(mining_indices, dtype=np.uint)[np.sort(
                np.random.RandomState(1).choice(len(mining_indices), size=args.mining_max_graphs, replace=False))]
        mining_graphs = [graphs[i] for i in mining_indices]
        if args.plot_mining_graphs:
            plot_graphs_to_dir(mining_graphs, os.path.join(args.output, 'problems'))
        if args.no_mining:
            save_df(utils.dataframe_from_records(records, 'problem', dtypes=dtypes), 'problems', args.output)
            return
        # TODO: Allow loading the subgraphs.
        subgraphs_with_support = mine_frequent_subgraphs(mining_graphs,
                                                         min_support=args.mining_min_support,
                                                         max_support=args.mining_max_support,
                                                         max_subgraphs_per_level=args.mining_max_subgraphs_per_level,
                                                         min_subgraph_size=args.mining_min_subgraph_size,
                                                         max_subgraph_size=args.mining_max_subgraph_size,
                                                         only_maximal=args.mining_only_maximal_subgraphs)
        save_df(utils.dataframe_from_records(
            [{'id': i,
              'edges': subgraph.size(),
              'nodes': len(subgraph),
              'support': support} for i, (subgraph, support) in
             enumerate(subgraphs_with_support)], 'id'), 'subgraphs', args.output)
        subgraphs_out_dir = os.path.join(args.output, 'subgraphs')
        os.makedirs(subgraphs_out_dir, exist_ok=True)
        for i, (subgraph, support) in enumerate(
                tqdm(subgraphs_with_support, unit='subgraph', desc='Saving frequent subgraphs')):
            nx.write_gpickle(subgraph, os.path.join(subgraphs_out_dir, f'{i}.gpickle'))
            if args.plot_frequent_subgraphs:
                out_file = os.path.join(subgraphs_out_dir, f'{i}.svg')
                plot_graph_to_file(subgraph, out_file)
        if args.no_featurization:
            save_df(utils.dataframe_from_records(records, 'problem', dtypes=dtypes), 'problems', args.output)
            return
        featurization_indices = all_graph_indices
        if args.featurization_max_graph_size is not None:
            featurization_indices = [i for i in all_graph_indices if
                                     graphs[i].size() <= args.featurization_max_graph_size]
        featurization_indices = np.asarray(featurization_indices, dtype=np.uint)
        logging.info('Problem graphs to featurize: %s', len(featurization_indices))
        featurization_graphs = [graphs[i] for i in featurization_indices]
        subgraphs = tuple(zip(*subgraphs_with_support))[0]
        feature_matrix = subgraph_matrix(featurization_graphs, subgraphs)
        for record_i, feature_vector in zip(featurization_indices, feature_matrix):
            records[record_i].update({('subgraph', k): int(feature_vector[k]) for k in range(len(feature_vector))})
        tfidf = TfidfTransformer(smooth_idf=False)
        feature_matrix = tfidf.fit_transform(feature_matrix)
        save_df(utils.dataframe_from_records(
            [{'id': i,
              'edges': subgraph.size(),
              'nodes': len(subgraph),
              'support': support,
              'idf': tfidf.idf_[i]} for i, (subgraph, support) in
             enumerate(subgraphs_with_support)], 'id'), 'subgraphs', args.output)
        subgraph_cols = [('subgraph', k) for k in range(len(subgraphs))]
        dtypes.update({col: pd.UInt8Dtype() for col in subgraph_cols})

        logging.info('Clustering...')
        center_indices = set()
        for metric in ('cosine', 'l1'):
            positions = {n: get_positions(n, metric, feature_matrix) for n in (10, 15, 20, 30, 40)}
            for n_clusters in (2, 3):
                kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, init='k-medoids++', random_state=0)
                labels = kmedoids.fit_predict(feature_matrix)
                center_indices.update(kmedoids.medoid_indices_)
                for label, i in enumerate(kmedoids.medoid_indices_):
                    distances = sklearn.metrics.pairwise_distances(kmedoids.cluster_centers_[label], feature_matrix,
                                                                   metric=metric).flatten()
                    second = np.argpartition(distances, 2)[1]
                    logging.info(
                        f'metric={metric}, n_clusters={n_clusters}, label={label}, representative={problems[featurization_indices[i]]}, second={problems[featurization_indices[second]]}')
                    center_indices.add(second)
                for local_i, (record_i, label) in enumerate(zip(featurization_indices, labels)):
                    records[record_i][('kmedoids', metric, n_clusters, 'label')] = label
                    for center_label, center_vector in enumerate(kmedoids.cluster_centers_):
                        records[record_i][('kmedoids', metric, n_clusters, 'center', center_label)] = int(
                            (feature_matrix[local_i] != center_vector).nnz == 0)
                dtypes[('kmedoids', metric, n_clusters, 'label')] = pd.UInt8Dtype()
                dtypes.update(
                    {('kmedoids', metric, n_clusters, 'center', center_label): pd.UInt8Dtype() for center_label in
                     range(n_clusters)})
                for depvar, (n_neighbors, pos) in itertools.product(
                        ('form', 'source', 'status', 'rating', 'spc', 'atoms', 'atoms_equality', 'equality_present'),
                        positions.items()):
                    plt.figure()
                    plt.title(f'metric={metric}, n_neighbors={n_neighbors}, n_clusters={n_clusters}, depvar={depvar}')
                    values = np.asarray([records[i][depvar] for i in featurization_indices])
                    if depvar in {'rating', 'atoms', 'atoms_equality'}:
                        depvar_map = values
                        cmap = 'copper'
                    else:
                        depvar_labels, depvar_map = np.unique(values, return_inverse=True)
                        cmap = 'Set1'
                    for label in range(n_clusters):
                        cluster_mask = labels == label
                        cur_pos = pos[cluster_mask]
                        c = depvar_map[cluster_mask]
                        marker = ['o', 's', '^'][label]
                        s = np.full(len(cur_pos), plt.rcParams['lines.markersize'] ** 2)
                        representative_mask = np.arange(len(labels)) == kmedoids.medoid_indices_[label]
                        s[representative_mask[cluster_mask]] = plt.rcParams['lines.markersize'] ** 2 * 4
                        plt.scatter(cur_pos[:, 0], cur_pos[:, 1], marker=marker, label=label, c=c, cmap=cmap, s=s)
                    plt.legend(title='Clusters')
                    out_file = os.path.join(args.output, 'manifolds',
                                            f'{metric}_{n_neighbors}_{depvar}_{n_clusters}.svg')
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    plt.savefig(out_file, bbox_inches='tight')
                    plt.close()
        plot_graphs_to_dir([graphs[featurization_indices[i]] for i in center_indices],
                           os.path.join(args.output, 'problems'))

        df = utils.dataframe_from_records(records, 'problem', dtypes=dtypes)
        save_df(df, 'problems', args.output)
        df_featurized = utils.dataframe_from_records((records[i] for i in featurization_indices), 'problem',
                                                     dtypes=dtypes)
        df_featurized.to_csv(os.path.join(args.output, 'vectors.tsv'), sep='\t', columns=subgraph_cols, header=False,
                             index=False)
        df_featurized.to_csv(os.path.join(args.output, 'metadata.tsv'), sep='\t',
                             columns=(col for col in col_dtypes.keys() if col != 'problem'))


def get_positions(n_neighbors, metric, feature_matrix):
    return Isomap(n_neighbors=n_neighbors, metric=metric, p=1).fit_transform(feature_matrix)


col_dtypes = {'problem': 'object',
              'clausify_returncode': pd.CategoricalDtype(),
              'clausify_time': np.float,
              'domain': pd.CategoricalDtype(),
              'number': pd.UInt16Dtype(),
              'form': pd.CategoricalDtype(),
              'version': pd.UInt8Dtype(),
              'size_parameters': 'object',
              'domain_full': pd.CategoricalDtype(),
              'source': pd.CategoricalDtype(),
              'status': pd.CategoricalDtype(),
              'rating': np.float,
              'spc': pd.CategoricalDtype(),
              'clauses': pd.UInt32Dtype(),
              'predicates': pd.UInt32Dtype(),
              'functions': pd.UInt32Dtype(),
              'graph_nodes': pd.UInt32Dtype(),
              'graph_edges': pd.UInt32Dtype(),
              'graph_time': np.float}


@memory.cache
def mine_frequent_subgraphs(graphs, min_support, max_support, max_subgraphs_per_level, min_subgraph_size,
                            max_subgraph_size, only_maximal):
    logging.info('Graphs to mine from: %s', len(graphs))
    logging.debug('Node counts: %s', [len(g) for g in graphs])
    logging.debug('Edge counts: %s', [g.size() for g in graphs])
    assert 0 <= min_support <= 1
    min_support = max(1, int(len(graphs) * min_support))
    if max_support is not None:
        assert 0 <= max_support <= 1
        max_support = int(len(graphs) * max_support)
    apr = Apriori(graphs,
                  min_support=min_support,
                  max_support=max_support,
                  max_width=max_subgraphs_per_level)
    return list(apr.frequent_subgraphs_with_support(max_k=max_subgraph_size, min_k=min_subgraph_size,
                                                    only_maximal=only_maximal))


@memory.cache
def subgraph_matrix(graphs, subgraphs):
    logging.info(
        f'Calculating incidences of {len(subgraphs)} subgraphs in {len(graphs)} graphs ({len(graphs) * len(subgraphs)} pairs)...')
    res = Parallel(verbose=1)(
        delayed(subgraph_is_isomorphic)(graph, subgraph) for graph, subgraph in itertools.product(graphs, subgraphs))
    return np.asarray(res, dtype=np.bool).reshape((len(graphs), len(subgraphs)))


class Apriori:
    """Edge-based join growth"""

    def __init__(self, graphs, min_support=1, max_support=None, max_width=None):
        self.graphs = sorted(graphs, key=lambda g: g.size(), reverse=True)
        self.min_support = min_support
        self.max_support = max_support
        self.max_width = max_width

    def frequent_subgraphs_with_support(self, max_k=None, min_k=0, only_maximal=True):
        assert min_k >= 0
        if max_k is None:
            ks = itertools.count(min_k)
            if self.min_support <= 0:
                warnings.warn('Will run forever.')
        else:
            ks = range(min_k, max_k + 1)
        for k in ks:
            gs = self.frequent_subgraphs_k_with_support(k)
            if len(gs) == 0:
                break
            if only_maximal and (max_k is None or k < max_k):
                incidence_matrix = subgraph_matrix(self.frequent_subgraphs_k(k + 1), [g[0] for g in gs])
                mask = np.sum(incidence_matrix, axis=0) == 0
            else:
                mask = np.ones(len(gs), dtype=np.bool)
            for i, g in enumerate(gs):
                assert g[1] >= self.min_support
                if g[1] <= self.max_support and mask[i]:
                    yield g

    @functools.lru_cache(maxsize=3)
    def frequent_subgraphs_k_with_support(self, k):
        assert k >= 0
        if k == 0:
            res = self.all_graphs_0()
        elif k == 1:
            res = self.all_graphs_1()
        else:
            buckets = collections.defaultdict(list)
            for core in tqdm(self.frequent_subgraphs_k(k - 2), unit='core', desc=f'Generating subgraphs for k = {k}'):
                assert core.size() == k - 2
                for g in self.join_pairs_of_graphs(core, self.frequent_subgraphs_k(k - 1)):
                    assert g.size() == k
                    signature = self.signature(g)
                    if any(is_isomorphic(other, g) for other in buckets[signature]):
                        continue
                    # TODO: Prune: Only add g if all of its subgraphs of size k-1 are present in
                    # frequent_subgraphs_k(k-1). Omit this pruning if self.max_support or self.max_width is not None.
                    buckets[signature].append(g)
            logging.debug(f'k = {k}: Non-empty buckets: {len(buckets)}')
            logging.debug(f'k = {k}: Average bucket size: {sum(len(v) for v in buckets.values()) / len(buckets)}')
            logging.debug(f'k = {k}: Maximum bucket size: {max(len(v) for v in buckets.values())}')
            # TODO: Remove graphs from `graphs` that support 0 subgraphs.
            res = list(itertools.chain.from_iterable(buckets.values()))
        logging.info(f'k = {k}: Candidate subgraphs: {len(res)}')
        res = self.filter_frequent_subgraphs(res)
        logging.info(f'k = {k}: Frequent subgraphs: {len(res)} / {self.max_width}')
        return res

    def frequent_subgraphs_k(self, k):
        return [g[0] for g in self.frequent_subgraphs_k_with_support(k)]

    @staticmethod
    def signature(g):
        return (len(g),
                tuple(sorted(d for n, d in g.in_degree())),
                tuple(sorted(d for n, d in g.out_degree())))

    @staticmethod
    def join_pairs_of_graphs(core, graphs):
        for i1 in range(len(graphs)):
            g1 = graphs[i1]
            assert g1.size() == core.size() + 1
            for iso1 in subgraph_isomorphisms(g1, core):
                assert set(core.nodes) == set(iso1.values())
                assert len(set(g1.nodes) - set(iso1.keys())) <= 1
                assert is_subgraph_isomorphism(g1, core, iso1)
                assert ('new', 0) not in iso1.values()
                g1_common = nx.relabel_nodes(g1, lambda x: iso1.get(x, ('new', 0)))
                assert set(core.edges) <= set(g1_common.edges)
                for i2 in range(i1, len(graphs)):
                    g2 = graphs[i2]
                    assert g2.size() == g1.size()
                    for iso2 in subgraph_isomorphisms(g2, core):
                        assert set(core.nodes) == set(iso2.values())
                        assert len(set(g2.nodes) - set(iso2.keys())) <= 1
                        assert is_subgraph_isomorphism(g2, core, iso2)
                        assert ('new', 1) not in iso2.values()
                        g2_common = nx.relabel_nodes(g2, lambda x: iso2.get(x, ('new', 1)))
                        assert set(core.edges) <= set(g2_common.edges)
                        g = nx.compose(g1_common, g2_common)
                        assert g.size() == core.size() + 2
                        yield nx.convert_node_labels_to_integers(g)
                        if g1_common.nodes[('new', 0)]['type'] == g2_common.nodes[('new', 1)]['type']:
                            g2_common = nx.relabel_nodes(g2, lambda x: iso2.get(x, ('new', 0)))
                            gj = nx.compose(g1_common, g2_common)
                            assert core.size() + 1 <= gj.size() <= core.size() + 2
                            if gj.size() == core.size() + 2:
                                yield nx.convert_node_labels_to_integers(gj)

    def all_graphs_1(self):
        subgraphs = []
        for type1, type2, attributes in edge_types:
            g = empty_graph()
            g.add_node(0, type=type1)
            g.add_node(1, type=type2)
            g.add_edge(0, 1, **attributes)
            assert all(not is_isomorphic(other, g) for other in subgraphs)
            assert g.size() == 1
            assert len(g) in {1, 2}
            subgraphs.append(g)
        return subgraphs

    def all_graphs_0(self):
        return [trivial_graph(node_type) for node_type in node_types]

    def filter_frequent_subgraphs(self, subgraphs):
        support_counts = self.get_supports(subgraphs)
        assert len(support_counts) == len(subgraphs)
        supports = collections.Counter(dict(zip(subgraphs, support_counts)))
        for subgraph, support in list(supports.items()):
            if support < self.min_support:
                del supports[subgraph]
        if len(supports) == 0:
            return []
        return supports.most_common(self.max_width)

    def get_supports(self, subgraphs):
        incidence_matrix = subgraph_matrix(self.graphs, subgraphs)
        logging.info('Graphs that support at least one subgraph: %d / %d',
                     np.count_nonzero(np.sum(incidence_matrix, axis=1)), len(self.graphs))
        return np.sum(incidence_matrix, axis=0)


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
    return DiGraphMatcher(g1, g2,
                          node_match=categorical_node_match('type', None),
                          edge_match=categorical_edge_match('polarity', None))


def trivial_graph(node_type):
    g = empty_graph()
    # We use the label "0" to mimic `nx.trivial_graph()`.
    g.add_node(0, type=node_type)
    return g


def empty_graph(name=None):
    return nx.DiGraph(name=name)


def problem_name(path):
    return path.replace('/', '_')


def problems_to_graphs(problems, solver, max_size=None):
    logging.info(f'Generating graph representations of {len(problems)} problems...')
    return Parallel(verbose=10)(delayed(problem_to_graph)(problem, solver, max_size) for problem in problems)


@memory.cache
def problem_to_graph(problem, solver, max_size=None):
    time_start = time.time()
    clausify_result = solver.clausify(problem)
    time_elapsed = time.time() - time_start
    record = {'problem': problem,
              'clausify_returncode': clausify_result.returncode,
              'clausify_time': time_elapsed}
    record.update(tptp.problem_properties(problem))
    if clausify_result.returncode != 0 or clausify_result.clauses is None or clausify_result.symbols is None:
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
    record.update({'graph_nodes': len(g), 'graph_edges': g.size(), 'graph_time': time_elapsed})
    if max_size is not None and g.size() > max_size:
        g = None
    return g, record


def clausify_result_to_graph(clausify_result, expression_namer, name=None):
    g = empty_graph(name=name)
    for symbol_type in ('predicate', 'function'):
        for i in range(len(clausify_result.symbols_of_type(symbol_type))):
            g.add_node(symbol_node_id(symbol_type, i), type=symbol_type, label=expression_namer.symbol(symbol_type, i))
    TermVisitor(g, expression_namer).visit_clauses(clausify_result.clauses)
    return g


def plot_graphs_to_dir(graphs, out_dir):
    logging.info(f'Plotting graphs to {out_dir}.')
    for g in tqdm(graphs, unit='graph', desc='Plotting graphs'):
        out_file = os.path.join(out_dir, f'{problem_name(g.name)}.svg')
        plot_graph_to_file(g, out_file)


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
    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    except ImportError:
        warnings.warn('Graphviz layout is not available.')
        pos = nx.spring_layout(g)
    for node_type, node_color in node_color_dict.items():
        nodelist = [node_id for node_id, nt in g.nodes.data('type') if nt == node_type]
        nx.draw_networkx_nodes(g, pos, nodelist=nodelist, node_color=node_color,
                               vmin=min(node_color_dict.values()), vmax=max(node_color_dict.values()), label=node_type)
    node_labels = None
    if custom_node_labels:
        node_labels = {k: node_label(v) for k, v in g.nodes.data()}
    nx.draw_networkx_labels(g, pos, font_size=8, labels=node_labels)
    edge_colors = [edge_color(e[2]) for e in g.edges(data=True)]
    nx.draw_networkx_edges(g, pos, edge_color=edge_colors)
    plt.legend()


node_types = [
    'root',
    'clause',
    'atom',
    'equality',
    'term',
    'argument',
    'predicate',
    'function',
    'variable'
]

edge_types = [('root', 'clause', {}),
              ('clause', 'atom', {'polarity': 0}),
              ('clause', 'atom', {'polarity': 1}),
              ('clause', 'equality', {'polarity': 0}),
              ('clause', 'equality', {'polarity': 1}),
              ('atom', 'predicate', {}),
              ('atom', 'argument', {}),
              ('equality', 'predicate', {}),
              ('equality', 'term', {}),
              ('equaltiy', 'variable', {}),
              ('term', 'argument', {}),
              ('term', 'function', {}),
              ('argument', 'term', {}),
              ('argument', 'variable', {})]

node_color_dict = {node_type: f'C{i}' for i, node_type in enumerate(node_types)}


def node_label(attributes):
    try:
        return attributes['label']
    except KeyError:
        return ''


def edge_color(attributes):
    if 'polarity' in attributes:
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
    main()
