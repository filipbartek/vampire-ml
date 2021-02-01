import collections
import functools
import hashlib
import json
import logging
import os
import time

import joblib
import pandas as pd
from joblib import Parallel, delayed

from proving import config
from proving.formula_visitor import FormulaVisitor
from proving.memory import memory
from proving.utils import py_str


@memory.cache(verbose=2)
def get_graphs(graphifier, problems):
    return graphifier.problems_to_graphs_dict(problems)


class Graphifier:
    """Stateless. Can be reused to create multiple graphs."""

    def __init__(self, clausifier, arg_order=True, arg_backedge=True, equality=True, max_number_of_nodes=None,
                 output_ntypes=('predicate', 'function')):
        self.clausifier = clausifier
        self.arg_order = arg_order
        self.arg_backedge = arg_backedge
        self.equality = equality
        self.max_number_of_nodes = max_number_of_nodes
        self.output_ntypes = output_ntypes
        self.version = 1

    @property
    def canonical_etypes(self):
        return self.empty_graph().canonical_etypes

    def problems_to_graphs_dict(self, problems):
        logging.info(f'Graphifying {len(problems)} problems...')
        graphs_records = Parallel(verbose=10)(delayed(self.problem_to_graph)(problem) for problem in problems)
        graphs, records = zip(*graphs_records)
        problem_graphs = {p: g for p, g, r in zip(problems, graphs, records) if
                          g is not None and r is not None and r['error'] is None}
        logging.info(f'Problems graphified. {len(problem_graphs)}/{len(graphs_records)} graphified successfully.')
        df = pd.DataFrame.from_records(records, index='problem')
        return problem_graphs, df

    def problem_to_graph(self, problem_name):
        if os.path.isabs(problem_name):
            cache_dir_full = os.path.join(self.cache_dir(), hashlib.md5(problem_name.encode()).hexdigest())
        else:
            cache_dir_full = os.path.join(self.cache_dir(), problem_name)
        filename_graph = os.path.join(cache_dir_full, 'graph.joblib')
        filename_record = os.path.join(cache_dir_full, 'record.json')
        try:
            with open(filename_record) as fp:
                record = json.load(fp)
            if record['error'] == 'clausify':
                if record['clausify_returncode'] is not None and record['clausify_returncode'] < 0:
                    raise RuntimeError('Clausification failed with negative return code: %d',
                                       record['clausify_returncode'])
                logging.debug(f'Skipping graphification of {problem_name} because its clausification failed.')
                graph = None
            elif record['graph_nodes_lower_bound'] > self.max_number_of_nodes:
                logging.debug(f'Skipping graphification of {problem_name} because it has at least %d nodes.',
                              record['graph_nodes_lower_bound'])
                graph = None
            else:
                assert 'graph_nodes' not in record or record['graph_nodes'] <= self.max_number_of_nodes
                # Raises ValueError if reading reaches an unexpected EOF.
                graph = joblib.load(filename_graph)
                logging.debug(f'Graph of {problem_name} loaded.')
                assert graph.num_nodes() == record['graph_nodes']
        except (FileNotFoundError, RuntimeError, ValueError):
            logging.debug(f'Failed to load graph of {problem_name}.', exc_info=True)
            graph, record = self.graphify(problem_name)
            os.makedirs(cache_dir_full, exist_ok=True)
            with open(filename_record, 'w') as fp:
                json.dump(record, fp, indent=4, default=int)
            if graph is not None:
                joblib.dump(graph, filename_graph)
        except Exception as e:
            raise RuntimeError(
                f'Failed to produce graph of problem {problem_name}. Graph file: {filename_graph}') from e
        assert graph is None or graph.num_nodes() <= self.max_number_of_nodes
        return graph, record

    @functools.lru_cache(maxsize=1)
    def cache_dir(self):
        cache_dir = os.path.join(config.cache_dir(), type(self).__name__, joblib.hash(self.get_config()))
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, 'config.json'), 'w') as fp:
            json.dump(self.get_config(), fp, indent=4, default=vars)
        return cache_dir

    @functools.lru_cache(maxsize=1)
    def get_config(self):
        attrs = (
        'version', 'clausifier', 'arg_order', 'arg_backedge', 'equality', 'max_number_of_nodes', 'output_ntypes')
        return {k: getattr(self, k) for k in attrs}

    def __repr__(self):
        return f'{self.__class__.__name__}(%s)' % ', '.join(f'{k}={v}' for k, v in self.get_config().items())

    def graphify(self, problem):
        problem = py_str(problem)
        logging.debug(f'Graphifying problem {problem}...')
        time_start = time.time()
        clausify_result = self.clausifier.clausify(problem)
        time_elapsed = time.time() - time_start
        record = {'problem': problem,
                  'clausify_returncode': clausify_result.returncode,
                  'clausify_time': time_elapsed,
                  'error': None}
        if clausify_result.returncode != 0 or clausify_result.clauses is None or clausify_result.symbols is None:
            logging.debug(f'Failed to graphify problem {problem}: clausification failed.')
            record['error'] = 'clausify'
            g = None
        else:
            symbol_types = ('predicate', 'function')
            symbols = {symbol_type: clausify_result.symbols_of_type(symbol_type) for symbol_type in symbol_types}
            record['num_clauses'] = len(clausify_result.clauses)
            record.update({f'num_{symbol_type}': len(symbols[symbol_type]) for symbol_type in symbol_types})
            time_start = time.time()
            try:
                g = self.clausify_result_to_graph(clausify_result)
            except FormulaVisitor.NumNodesError as e:
                # The graph would be too large (too many nodes).
                logging.debug(f'Failed to graphify problem {problem}.', exc_info=True)
                record['error'] = 'node_count'
                g = None
                record['graph_nodes_lower_bound'] = e.actual
            time_elapsed = time.time() - time_start
            record['graph_time'] = time_elapsed
            if g is not None:
                record['graph_nodes'] = g.num_nodes()
                record['graph_nodes_lower_bound'] = g.num_nodes()
                record.update({f'graph_nodes_{ntype}': g.num_nodes(ntype) for ntype in g.ntypes})
                record['graph_edges'] = sum(g.num_edges(canonical_etype) for canonical_etype in g.canonical_etypes)
                logging.debug(f'Problem {problem} graphified.')
        return g, record

    @functools.lru_cache(maxsize=1)
    def empty_graph(self):
        output_ntypes = {ntype: 0 for ntype in self.output_ntypes}
        return self.formula_visitor().graph(output_ntypes)

    def clausify_result_to_graph(self, clausify_result):
        visitor = self.formula_visitor()
        visitor.visit_formula(clausify_result.clauses)
        symbol_features = {symbol_type: clausify_result.symbols_of_type(symbol_type)[self.symbol_feature_columns] for
                           symbol_type in self.symbol_types}
        output_ntypes = {ntype: len(symbol_features[ntype]) for ntype in self.output_ntypes}
        return visitor.graph(output_ntypes, node_features=symbol_features)

    def formula_visitor(self):
        return FormulaVisitor(arg_order=self.arg_order, arg_backedge=self.arg_backedge, equality=self.equality,
                              max_number_of_nodes=self.max_number_of_nodes)

    symbol_types = ('predicate', 'function')
    symbol_feature_columns = ['inGoal', 'introduced']

    @property
    def ntype_in_degrees(self):
        dtype_etypes = collections.defaultdict(set)
        for stype, etype, dtype in self.canonical_etypes:
            dtype_etypes[dtype].add((stype, etype))
        return {dtype: len(etypes) for dtype, etypes in dtype_etypes.items()}

    @property
    def ntype_feat_sizes(self):
        return {
            'clause': FormulaVisitor.num_clause_roles,
            **{symbol_type: len(self.symbol_feature_columns) for symbol_type in self.symbol_types}
        }
