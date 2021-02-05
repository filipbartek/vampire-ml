import itertools
import os
import warnings

import joblib
import more_itertools
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf

from proving import vampire
from proving.utils import dataframe_from_records
from proving.utils import py_str
from vampire_ml.results import save_df


class SymbolCostEvaluation(tf.keras.callbacks.CSVLogger):
    name = 'solver_eval'

    def __init__(self, cfg, csv_filename, solver, problems, symbol_type, splits, tensorboard=None,
                 problem_categories=None, parallel=None, baseline=False, **kwargs):
        super().__init__(csv_filename, **kwargs)
        self.solver = solver
        self.problems = problems
        self.problems_dataset = tf.data.Dataset.from_tensor_slices(problems).batch(cfg.batch_size)
        self.symbol_type = symbol_type
        start = cfg.start
        step = cfg.step
        if start is None and step is not None:
            start = -1
        elif step is None and start is not None:
            step = 1
        self.start = start
        self.step = step
        self.iterations = cfg.iterations
        self.tensorboard = tensorboard
        self.splits = splits  # validation, train
        if problem_categories is None:
            problem_categories = {'all': None}
        self.problem_categories = problem_categories
        if parallel is None:
            parallel = joblib.Parallel(verbose=10)
        self.parallel = parallel
        self.cache = cfg.cache
        self.baseline = baseline
        self.train_without_questions = cfg.train_without_questions

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

        if not self.baseline:
            precedences, precedence_costs = self.precedences(symbol_costs)
        else:
            precedences = itertools.repeat(None, len(problems))

        print(f'Evaluating with solver on {tf.shape(problems)[0]} problems {self.iterations} times...')
        if self.cache and self.iterations > 1:
            warnings.warn(
                f'Evaluating with caching and {self.iterations} iterations. Evaluating more than once only makes sense without caching.')
        records = self.parallel(joblib.delayed(self.solve_one)(problem, precedence) for problem, precedence in
                                more_itertools.ncycles(zip(problems, precedences), self.iterations))

        problems_list = list(map(py_str, problems))
        df_data = {'problem': problems_list}
        if not self.baseline:
            df_data.update({'symbols': symbol_costs.row_lengths(), 'precedence_cost': precedence_costs})
        iter_dfs = []
        keys = ['returncode', 'time_elapsed', 'time_elapsed_vampire', 'saturation_iterations', 'memory_used']
        field_series = {k: [] for k in keys}
        for i in range(self.iterations):
            iter_records = records[i * len(problems):(i + 1) * len(problems)]
            iter_dicts = [{k: getattr(r, k) for k in keys} for r in iter_records]
            iter_df = dataframe_from_records(iter_dicts, dtypes=vampire.Result.pd_dtypes, index=problems_list)
            iter_dfs.append(iter_df)
            for k in keys:
                field_series[k].append(iter_df[k])
        for k, l in field_series.items():
            df = pd.DataFrame(dict(enumerate(l)))
            if k == 'returncode':
                df_data['success', 'rate'] = (df == 0).mean(axis=1)
            else:
                df_data.update({
                    (k, 'mean'): df.mean(axis=1),
                    (k, 'std'): np.std(df, axis=1),
                    (k, 'variation'): scipy.stats.variation(df.to_numpy(dtype=np.float, na_value=np.nan), axis=1)
                })
                if k == 'memory_used':
                    df_data[k, 'max'] = df.max(axis=1)
        main_iter_df = pd.concat(iter_dfs, axis='columns', keys=range(self.iterations))
        header_df = pd.DataFrame(df_data)
        header_df.set_index('problem', inplace=True)
        main_df = pd.concat([header_df, main_iter_df], axis='columns')
        main_df.index.name = 'problem'
        output_dir = os.path.join(self.name, symbol_cost_model.name, f'epoch_{epoch}')
        os.makedirs(output_dir, exist_ok=True)

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
                    df_success = records_df[[(i, 'returncode') for i in range(self.iterations)]] == 0
                    if cat_problems is None:
                        count_all = len(set(dataset_problems))
                        count_filtered = len(set(self.problems) & set(dataset_problems))
                    else:
                        count_all = len(set(dataset_problems) & set(cat_problems))
                        count_filtered = len(set(self.problems) & set(dataset_problems) & set(cat_problems))
                    res = {
                        'problem/count/split': count_all,
                        'problem/count/eval': count_filtered,
                        'problem/count/valid': len(records_df),
                        'success/count/mean': df_success.sum(axis=0).mean(),
                        'success/count/std': np.std(df_success.sum(axis=0)),
                        'success/rate/mean': df_success.mean(axis=0).mean()
                    }
                    for k, v in res.items():
                        logs[f'{dataset_name}/{cat_name}/{k}'] = v
                        tf.summary.scalar(f'{summary_name}/{k}', v, step=epoch)
                    if not self.baseline:
                        for column_name in ['symbols', 'precedence_cost']:
                            values = records_df[column_name]
                            tf.summary.histogram(f'{summary_name}/{column_name}', values, step=epoch)
                    for column_name in ['time_elapsed', 'saturation_iterations']:
                        values = records_df[[(i, column_name) for i in range(self.iterations)]].astype(float)
                        tf.summary.histogram(f'{summary_name}/{column_name}/all', values, step=epoch)
        save_df(main_df, os.path.join(output_dir, 'problems'))

        with self.tensorboard.train_writer.as_default():
            tf.summary.scalar('time/epoch/solver_eval', tf.timestamp() - time_begin, step=epoch)

        return logs

    def precedences(self, symbol_costs):
        precedences, precedence_costs = zip(*map(self.precedence, symbol_costs))
        precedence_costs = tf.stack(precedence_costs)
        return precedences, precedence_costs

    def precedence(self, symbol_cost):
        # We sort the symbols by cost in non-decreasing order.
        precedence = tf.argsort(symbol_cost, direction='DESCENDING')
        n_symbols = symbol_cost.shape[0]
        precedence_cost = tf.tensordot(tf.gather(symbol_cost, precedence),
                                       tf.range(n_symbols, dtype=symbol_cost.dtype), 1) * 2 / (
                                      n_symbols * (n_symbols + 1))
        return precedence, precedence_cost

    def solve_one(self, problem, precedence=None):
        precedences = None
        if precedence is not None:
            precedences = {self.symbol_type: precedence.numpy()}
        return self.solver.solve(py_str(problem), precedences=precedences, cache=self.cache)
