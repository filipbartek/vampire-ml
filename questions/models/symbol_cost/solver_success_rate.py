import pandas as pd
import tensorflow as tf

from proving import vampire
from proving.utils import dataframe_from_records


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
        if valid is not None:
            problems = tf.boolean_mask(problems, valid)
            symbol_costs = tf.ragged.boolean_mask(symbol_costs, valid)
            values = tf.map_fn(self.evaluate_one, (problems, symbol_costs), fn_output_signature=tf.bool)
        else:
            values = tf.map_fn(self.evaluate_one_baseline, problems, fn_output_signature=tf.bool)
        super().update_state(values, **kwargs)

    def evaluate_one(self, x):
        problem, symbol_cost = x
        precedence = tf.argsort(symbol_cost)
        return tf.py_function(self.solve_one, [problem, precedence], tf.bool)

    def evaluate_one_baseline(self, problem):
        return tf.py_function(self.solve_one, [problem], tf.bool)

    def solve_one(self, problem, precedence=None):
        problem = bytes.decode(problem.numpy())
        record = {'problem': problem, 'valid': True}
        precedences = None
        if precedence is not None:
            precedence = precedence.numpy()
            precedences = {self.symbol_type: precedence}
            record['n_symbols'] = len(precedence)
        solver_res = self.solver.solve(problem, precedences=precedences)
        record.update(solver_res.as_record())
        self.records.append(record)
        return solver_res.returncode == 0

    def result_df(self):
        return dataframe_from_records(self.records, index_keys='problem', dtypes=self.record_pd_dtypes)
