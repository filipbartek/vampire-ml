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

    def __init__(self, solver, symbol_type, parallel=None, baseline=False, name='solver_success_rate', **kwargs):
        super().__init__(name=name, **kwargs)
        self.solver = solver
        self.symbol_type = symbol_type
        if parallel is None:
            parallel = Parallel(verbose=1)
        self.parallel = parallel
        self.baseline = baseline
        self.records = []

    def reset_states(self):
        super().reset_states()
        self.records = []

    def update_state(self, problems, symbol_costs, valid, **kwargs):
        # Ignore the invalid symbol costs. Only measure performance on the valid symbol costs.
        problems = tf.boolean_mask(problems, valid)
        symbol_costs = tf.ragged.boolean_mask(symbol_costs, valid)
        values = tf.py_function(self.solve_multiple, [problems, symbol_costs.flat_values, symbol_costs.row_splits],
                                tf.bool)
        super().update_state(values, **kwargs)

    def solve_multiple(self, problems, flat_values, row_splits):
        symbol_costs = tf.RaggedTensor.from_row_splits(flat_values, row_splits)
        print(f'Evaluating on a batch of {tf.shape(problems)[0]} problems...')
        records = self.parallel(
            delayed(self.solve_one)(problem, symbol_cost) for problem, symbol_cost in zip(problems, symbol_costs))
        self.records.extend(records)
        values = [r['returncode'] == 0 for r in records]
        values = tf.convert_to_tensor(values)
        return values

    def solve_one(self, problem, symbol_cost):
        problem = py_str(problem)
        record = {'problem': problem}
        precedences = None
        if not self.baseline:
            # We sort the symbols by cost in non-decreasing order.
            precedence = tf.argsort(symbol_cost)
            precedence = precedence.numpy()
            precedences = {self.symbol_type: precedence}
            record['n_symbols'] = len(precedence)
        solver_res = self.solver.solve(problem, precedences=precedences, cache=False)
        record.update(solver_res.as_record())
        return record

    def result_df(self):
        return dataframe_from_records(self.records, index_keys='problem', dtypes=self.record_pd_dtypes)
