import itertools

import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed

from proving import vampire
from proving.utils import dataframe_from_records
from proving.utils import py_str


class ValidityRate(tf.keras.metrics.Mean):
    def __init__(self, name='validity_rate', **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, problems, symbol_costs, valid, **kwargs):
        super().update_state(valid, **kwargs)


class SolverSuccessRate(tf.keras.metrics.Mean):
    record_pd_dtypes = {
        'problem': 'object',
        'valid': bool,
        'n_symbols': pd.UInt32Dtype(),
        **vampire.Result.pd_dtypes
    }

    def __init__(self, solver, symbol_type, name='solver_success_rate', **kwargs):
        super().__init__(name=name, **kwargs)
        self.solver = solver
        self.symbol_type = symbol_type
        self.records = []

    def reset_states(self):
        super().reset_states()
        self.records = []

    def update_state(self, problems, symbol_costs=None, valid=None, **kwargs):
        if len(problems.shape) == 2:
            problems = tf.squeeze(problems, axis=1)
        if symbol_costs is not None and valid is not None:
            # Ignore the invalid symbol costs. Only measure performance on the valid symbol costs.
            problems = tf.boolean_mask(problems, valid)
            symbol_costs = tf.ragged.boolean_mask(symbol_costs, valid)
            values = tf.py_function(self.solve_multiple, [problems, symbol_costs.flat_values, symbol_costs.row_splits],
                                    tf.bool)
        else:
            # Evaluate the baseline solver performance
            values = tf.py_function(self.solve_multiple, [problems], tf.bool)
        super().update_state(values, **kwargs)

    def solve_multiple(self, problems, flat_values=None, row_splits=None):
        if flat_values is None or row_splits is None:
            symbol_costs = itertools.repeat(None)
        else:
            symbol_costs = tf.RaggedTensor.from_row_splits(flat_values, row_splits)
        records = Parallel(verbose=0)(
            delayed(self.solve_one)(problem, symbol_cost) for problem, symbol_cost in zip(problems, symbol_costs))
        self.records.extend(records)
        return [r['returncode'] == 0 for r in records]

    def solve_one(self, problem, symbol_cost=None):
        problem = py_str(problem)
        record = {'problem': problem, 'valid': symbol_cost is not None}
        precedences = None
        if symbol_cost is not None:
            # We sort the symbols by cost in non-decreasing order.
            precedence = tf.argsort(symbol_cost)
            precedence = precedence.numpy()
            precedences = {self.symbol_type: precedence}
            record['n_symbols'] = len(precedence)
        solver_res = self.solver.solve(problem, precedences=precedences)
        record.update(solver_res.as_record())
        return record

    def result_df(self):
        return dataframe_from_records(self.records, index_keys='problem', dtypes=self.record_pd_dtypes)
