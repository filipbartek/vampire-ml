import collections
import functools
import hashlib
import json
import logging
import os
import sys

import joblib
import pandas as pd
import yaml
from joblib import Parallel, delayed

from questions import config
from questions import tptp
from questions.formula_visitor import FormulaVisitor
from questions.memory import memory
from questions.utils import dataframe_from_records
from questions.utils import CsvDictWriter
from questions.utils import py_str
from questions.utils import timer
from utils import get_parallel
from weight import yaml_utils


@memory.cache(verbose=1, ignore=['expensive'])
def problems_to_graphs_list(graphifier, problems, expensive=True, return_graphs=True):
    return graphifier.compute_graphs(problems, expensive=expensive, return_graphs=return_graphs)


class Graphifier:
    """Stateless. Can be reused to create multiple graphs."""

    ignore = ['writer']

    def __init__(self, clausifier, arg_order=True, arg_backedge=True, equality=True, max_number_of_nodes=None,
                 output_ntypes=('predicate', 'function'), extra_fieldnames=None):
        self.clausifier = clausifier
        self.arg_order = arg_order
        self.arg_backedge = arg_backedge
        self.equality = equality
        self.max_number_of_nodes = max_number_of_nodes
        self.output_ntypes = output_ntypes
        self.version = 1
        fieldnames = {'attempts': None,
                      'total': {k: None for k in ['graph_nodes', 'graph_edges']},
                      'error': {k: None for k in [None, 'node_count', 'nodes_from_tptp_header', 'recursion']},
                      'clausify_cached': {k: None for k in [False, True]},
                      'expensive': None}
        if extra_fieldnames is not None:
            fieldnames.update({k: None for k in extra_fieldnames})
        self.writer = CsvDictWriter(open('graphifier.csv', 'w'), fieldnames, sep='/')
        self.writer.writeheader()

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self.ignore}

    @property
    def canonical_etypes(self):
        return self.empty_graph().canonical_etypes

    def get_graphs_dict(self, problems, cache=True):
        graphs, df = self.get_graphs(problems, cache=cache, get_df=True)
        records = (x[1] for x in df.iterrows())
        problem_graphs = {p: g for p, g, r in zip(problems, graphs, records) if
                          g is not None and r is not None and r['error'] is None}
        logging.info(f'Problems graphified. {len(problem_graphs)}/{len(df)} graphified successfully.')
        return problem_graphs, df

    def get_graphs(self, problems, expensive=True, get_df=False, return_graphs=True, write_row=True, **kwargs):
        if expensive:
            # Cache the result
            graphs_records = problems_to_graphs_list(self, problems, expensive=expensive, return_graphs=return_graphs)
        else:
            graphs_records = self.compute_graphs(problems, expensive=expensive)
        graphs, records = zip(*graphs_records)
        df = dataframe_from_records(records, index='problem', dtypes=self.dtypes())
        for col in ['graph_nodes', 'graph_edges', 'error', 'clausify_cached']:
            if col not in df:
                df[col] = pd.Series([pd.NA] * len(df), dtype=self.dtypes().get(col))
        stats = {
            'attempts': len(df),
            'total': df[['graph_nodes', 'graph_edges']].sum().to_dict(),
            'error': df['error'].value_counts(dropna=False).to_dict(),
            'clausify_cached': df['clausify_cached'].value_counts().to_dict(),
            'expensive': expensive,
            **kwargs
        }
        if write_row:
            self.writer.writerow(stats)
        logging.debug(f'Problems converted to graphs.\n%s\n%s' % (yaml.dump(stats), df[['graph_nodes', 'graph_edges', 'graph_nodes_lower_bound', 'error', 'clausify_cached']]))
        if get_df:
            return graphs, df
        else:
            return graphs

    def dtypes(self):
        return {
            'clausify_returncode': pd.Int8Dtype(),
            'num_clauses': pd.UInt32Dtype(),
            'num_predicate': pd.UInt32Dtype(),
            'num_function': pd.UInt32Dtype(),
            'graph_nodes': pd.UInt32Dtype(),
            'graph_nodes_lower_bound': pd.UInt32Dtype(),
            **{f'graph_nodes_{ntype}': pd.UInt32Dtype() for ntype in self.formula_visitor().ntypes()},
            'graph_edges': pd.UInt32Dtype()
        }

    def compute_graphs(self, problems, expensive=True, return_graphs=True):
        if expensive and len(problems) > 1:
            print(f'Graphifying {len(problems)} problems of at most {self.max_number_of_nodes} nodes...',
                  file=sys.stderr)
            n_jobs = None
            verbose = 10
        else:
            n_jobs = 1
            verbose = 0
        with get_parallel(len(problems), n_jobs=n_jobs, verbose=verbose) as parallel:
            logging.debug('Graphifying. Parent: CUDA_VISIBLE_DEVICES=%s' % os.environ['CUDA_VISIBLE_DEVICES'])
            return parallel(delayed(self.problem_to_graph)(problem, return_graph=return_graphs) for problem in problems)

    def problem_to_graph(self, problem_name, cache=True, return_graph=True):
        logging.debug('Graphifying. Worker: CUDA_VISIBLE_DEVICES=%s' % os.environ['CUDA_VISIBLE_DEVICES'])
        graph = None
        record = None
        if os.path.isabs(problem_name):
            cache_dir_full = os.path.join(self.cache_dir(), hashlib.md5(problem_name.encode()).hexdigest())
        else:
            cache_dir_full = os.path.join(self.cache_dir(), problem_name)
        filename_graph = os.path.join(cache_dir_full, 'graph.joblib')
        filename_record = os.path.join(cache_dir_full, 'record.json')
        graph_instantiated = False
        if cache:
            try:
                with open(filename_record) as fp:
                    record = json.load(fp)
                if record['error'] == 'clausify':
                    if record['clausify_returncode'] is not None and record['clausify_returncode'] < 0:
                        raise RuntimeError('Clausification failed with negative return code: %d',
                                           record['clausify_returncode'])
                    logging.debug(f'Skipping graphification of {problem_name} because its clausification failed.')
                    graph = None
                elif self.max_number_of_nodes is not None and record[
                    'graph_nodes_lower_bound'] > self.max_number_of_nodes:
                    logging.debug(f'Skipping graphification of {problem_name} because it has at least %d nodes.',
                                  record['graph_nodes_lower_bound'])
                    graph = None
                else:
                    assert 'graph_nodes' not in record or self.max_number_of_nodes is None or record[
                        'graph_nodes'] <= self.max_number_of_nodes
                    # Raises EOFError or ValueError if reading reaches an unexpected EOF.
                    graph = joblib.load(filename_graph)
                    assert graph.num_nodes() == record['graph_nodes']
                graph_instantiated = True
            except (FileNotFoundError, RuntimeError, ValueError, EOFError, KeyError):
                logging.debug(f'Failed to load graph of {problem_name}.', exc_info=True)
            except Exception as e:
                raise RuntimeError(
                    f'Failed to produce graph of problem {problem_name}. Graph file: {filename_graph}') from e
        if not graph_instantiated:
            graph, record = self.graphify(problem_name, cache=cache)
            if cache:
                os.makedirs(cache_dir_full, exist_ok=True)
                with open(filename_record, 'w') as fp:
                    json.dump(record, fp, indent=4, default=int)
                if graph is not None:
                    joblib.dump(graph, filename_graph)
        assert graph is None or self.max_number_of_nodes is None or graph.num_nodes() <= self.max_number_of_nodes
        record['graph_loaded_from_cache'] = graph_instantiated
        device = None
        if graph is not None:
            device = graph.device
        logging.debug(f'Graphification of {problem_name} finished. Loaded from cache: {graph_instantiated}. Device: {device}.')
        if not return_graph:
            graph = None
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

    def nodes_lower_bound(self, problem):
        props = tptp.problem_properties(problem)
        prop_names = ['atoms', 'predicates', 'functors', 'variables']
        return sum(props.get(k, 0) for k in prop_names)

    def graphify(self, problem, cache=True):
        # TODO: Time the whole call (all inclusive).
        problem = py_str(problem)
        logging.debug(f'Graphifying problem {problem}...')
        record = {'problem': problem, 'error': None, 'clausify_cached': cache}
        nodes_lower_bound = self.nodes_lower_bound(problem)
        if self.max_number_of_nodes is not None and nodes_lower_bound > self.max_number_of_nodes:
            record['error'] = 'nodes_from_tptp_header'
            record['graph_nodes_lower_bound'] = nodes_lower_bound
            return None, record
        with timer() as t:
            clausify_result = self.clausifier.clausify(problem, cache=cache)
        record.update({'clausify_returncode': clausify_result.returncode,
                       'clausify_time': t.elapsed,
                       'clausify_time_original': clausify_result.time_elapsed})
        if clausify_result.returncode != 0 or clausify_result.clauses is None or clausify_result.symbols is None:
            logging.debug(f'Failed to graphify problem {problem}: clausification failed.')
            record['error'] = 'clausify'
            return None, record
        symbol_types = ('predicate', 'function')
        symbols = {symbol_type: clausify_result.symbols_of_type(symbol_type) for symbol_type in symbol_types}
        record['num_clauses'] = len(clausify_result.clauses)
        record.update({f'num_{symbol_type}': len(symbols[symbol_type]) for symbol_type in symbol_types})
        g = None
        with timer() as t:
            try:
                g = self.clausify_result_to_graph(clausify_result)
            except FormulaVisitor.NumNodesError as e:
                # The graph would be too large (too many nodes).
                logging.debug(f'Failed to graphify problem {problem}.', exc_info=True)
                record['error'] = 'node_count'
                record['graph_nodes_lower_bound'] = e.actual
            except RecursionError:
                logging.warning(f'Failed to graphify problem {problem}.', exc_info=True)
                record['error'] = 'recursion'
                record['graph_nodes_lower_bound'] = 0
        record['graph_time'] = t.elapsed
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

    def clausify_result_to_graph(self, clausify_result, **kwargs):
        visitor = self.formula_visitor()
        visitor.visit_formula(clausify_result.clauses)
        symbol_features = {symbol_type: clausify_result.symbols_of_type(symbol_type)[self.symbol_feature_columns] for
                           symbol_type in self.symbol_types}
        output_ntypes = {ntype: len(symbol_features[ntype]) for ntype in self.output_ntypes}
        return visitor.graph(output_ntypes, node_features=symbol_features, **kwargs)

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
