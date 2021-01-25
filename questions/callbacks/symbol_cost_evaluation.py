import os

import joblib
import pandas as pd
import tensorflow as tf

from proving import vampire
from proving.utils import dataframe_from_records
from proving.utils import py_str
from vampire_ml.results import save_df


class SymbolCostEvaluation(tf.keras.callbacks.CSVLogger):
    name = 'solver_eval'

    def __init__(self, csv_filename, solver, problems, symbol_type, splits, batch_size=1, start=-1, step=1,
                 output_dir=None, tensorboard=None,
                 problem_categories=None, parallel=None, baseline=False, train_without_questions=False, **kwargs):
        super().__init__(csv_filename, **kwargs)
        self.solver = solver
        self.problems = problems
        self.problems_dataset = tf.data.Dataset.from_tensor_slices(problems).batch(batch_size)
        self.symbol_type = symbol_type
        if start is None and step is not None:
            start = -1
        elif step is None and start is not None:
            step = 1
        self.start = start
        self.step = step
        self.output_dir = output_dir
        self.tensorboard = tensorboard
        self.splits = splits  # validation, train
        if problem_categories is None:
            problem_categories = {'all': None}
        self.problem_categories = problem_categories
        if parallel is None:
            parallel = joblib.Parallel(verbose=10)
        self.parallel = parallel
        self.baseline = baseline
        self.train_without_questions = train_without_questions

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.start is not None and self.step is not None and epoch >= self.start and (
                epoch - self.start) % self.step == 0:
            logs.update(self.evaluate(self.model.symbol_cost_model, epoch))
            print(f'Metrics after epoch {epoch}: {logs}')
            super().on_epoch_end(epoch, logs=logs)

    def evaluate(self, symbol_cost_model, epoch=None):
        time_begin = tf.timestamp()

        logs = {}
        if len(self.problems) == 0:
            return logs
        print(f'Evaluating symbol cost model \'{symbol_cost_model.name}\' on {len(self.problems)} problems...')
        print(f'Predicting symbol costs on {len(self.problems)} problems...')
        res = symbol_cost_model.predict(self.problems_dataset, verbose=1)

        symbol_costs = res['costs']
        valid = res['valid']

        # Ignore the invalid symbol costs. Only measure performance on the valid symbol costs.
        problems = tf.boolean_mask(self.problems, valid)
        symbol_costs = tf.ragged.boolean_mask(symbol_costs, valid)
        print(f'Evaluating with solver on {tf.shape(problems)[0]} problems...')
        records = self.parallel(joblib.delayed(self.solve_one)(problem, symbol_cost) for problem, symbol_cost in
                                zip(problems, symbol_costs))
        main_df = dataframe_from_records(records, index_keys='problem', dtypes=self.record_pd_dtypes)
        for dataset_name, dataset_problems in self.splits.items():
            df_dataset = main_df.loc[main_df.index.intersection(dataset_problems)]
            with self.tensorboard.writers[dataset_name].as_default():
                for cat_name, cat_problems in self.problem_categories.items():
                    if dataset_name == 'train' and cat_name == 'all' and not self.train_without_questions:
                        continue
                    summary_name = f'{self.name}/{cat_name}'
                    if cat_problems is None:
                        records_df = df_dataset
                    else:
                        records_df = df_dataset.loc[df_dataset.index.intersection(cat_problems)]
                    res = {
                        'problem/count': len(records_df),
                        'success/count': (records_df['returncode'] == 0).sum(),
                        'success/rate': (records_df['returncode'] == 0).mean(),
                    }
                    for k, v in res.items():
                        logs[f'{dataset_name}/{cat_name}/{k}'] = v
                        tf.summary.scalar(f'{summary_name}/{k}', v, step=epoch)
                    for column_name in ['time_elapsed', 'saturation_iterations']:
                        values = records_df[column_name].astype(float)
                        tf.summary.histogram(f'{summary_name}/{column_name}/all', values, step=epoch)
                        tf.summary.histogram(f'{summary_name}/{column_name}/succ',
                                             values[records_df['returncode'] == 0], step=epoch)
                        tf.summary.histogram(f'{summary_name}/{column_name}/fail',
                                             values[records_df['returncode'] != 0], step=epoch)
        if self.output_dir is not None:
            output_dir = os.path.join(self.output_dir, self.name, symbol_cost_model.name, f'epoch_{epoch}')
            save_df(main_df, os.path.join(output_dir, 'problems'))

        with self.tensorboard.train_writer.as_default():
            tf.summary.scalar('time/epoch/solver_eval', tf.timestamp() - time_begin, step=epoch)

        return logs

    def solve_one(self, problem, symbol_cost):
        problem = py_str(problem)
        n_symbols = symbol_cost.shape[0]
        record = {'problem': problem, 'n_symbols': n_symbols}
        precedences = None
        if not self.baseline:
            # We sort the symbols by cost in non-decreasing order.
            precedence = tf.argsort(symbol_cost, direction='DESCENDING')
            precedence_cost = tf.tensordot(tf.gather(symbol_cost, precedence),
                                           tf.range(n_symbols, dtype=symbol_cost.dtype), 1) * 2 / (
                                          n_symbols * (n_symbols + 1))
            record['precedence_cost'] = precedence_cost.numpy()
            precedence = precedence.numpy()
            precedences = {self.symbol_type: precedence}
        solver_res = self.solver.solve(problem, precedences=precedences, cache=False)
        record.update(solver_res.as_record())
        return record

    record_pd_dtypes = {
        'problem': 'object',
        'valid': bool,
        'n_symbols': pd.UInt32Dtype(),
        **vampire.Result.pd_dtypes
    }

    @staticmethod
    def log_key(dataset_name, metric_name):
        assert dataset_name in {'train', 'val'}
        if dataset_name == 'val':
            return f'val_{metric_name}'
        return metric_name
