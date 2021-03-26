import itertools
import os
import sys
import warnings

import joblib
import more_itertools
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import yaml

from questions.results import save_df
from questions.solver import vampire
from questions.utils import dataframe_from_records
from questions.utils import flatten_dict
from questions.utils import timer


class SymbolCostEvaluation(tf.keras.callbacks.CSVLogger):
    name = 'solver_eval'
    columns = ['returncode', 'time_elapsed', 'time_elapsed_vampire', 'saturation_iterations', 'memory_used']

    def __init__(self, cfg, csv_filename, solver, problems, splits, symbol_type, tensorboard=None,
                 problem_categories=None, parallel=None, baseline=False, **kwargs):
        super().__init__(csv_filename, **kwargs)
        self.solver = solver
        self.problems = problems
        self.problems_dataset = tf.data.Dataset.from_tensor_slices(problems).batch(cfg.batch_size)
        start = cfg.start
        step = cfg.step
        if start is None and step is not None:
            start = -1
        elif step is None and start is not None:
            step = 1
        self.start = start
        self.step = step
        self.iterations = cfg.iterations
        self.symbol_type = symbol_type
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
        self.isolated = cfg.isolated

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.start is not None and self.step is not None and epoch >= self.start and (
                epoch - self.start) % self.step == 0:
            logs.update(self.flatten_logs(self.evaluate({self.symbol_type: self.model.symbol_cost_model}, epoch)))
            super().on_epoch_end(epoch, logs=logs)

    @staticmethod
    def flatten_logs(logs):
        return flatten_dict(logs, sep='/')

    def evaluate(self, models, epoch=None):
        if self.cache and self.iterations > 1:
            warnings.warn(
                f'Evaluating with caching and {self.iterations} iterations. Evaluating more than once only makes sense without caching.')

        if len(self.problems) == 0:
            # If there are no problems, return an empty log.
            return {}

        time_begin = tf.timestamp()

        print(
            f'Evaluating symbol cost model with solver on {len(self.problems)} problems after epoch {epoch}...',
            file=sys.stderr)

        df_data = {'problem': self.problems}

        get_precedences = None
        if not self.isolated and not self.baseline:
            data = {}
            for name, model in models.items():
                print(f'Predicting {name} symbol costs on {len(self.problems)} problems...')
                res = model.predict(self.problems_dataset, verbose=1)
                precedences, precedence_costs = self.precedences(res['costs'])
                data[name] = {
                    'valid': res['valid'],
                    'precedences': precedences
                }
                df_data[name, 'symbols'] = res['costs'].row_lengths()
                df_data[name, 'symbol_cost_valid'] = res['valid']
                df_data[name, 'precedence_cost'] = precedence_costs

            def get_precedences(problem_i):
                res = {}
                for name, vals in data.items():
                    if not vals['valid'][problem_i]:
                        return None
                    res[name] = vals['precedences'][problem_i]
                return res

        cases = more_itertools.ncycles(range(len(self.problems)), self.iterations)
        print(
            f'Evaluating on {len(self.problems) * self.iterations} cases ({len(self.problems)} problems, {self.iterations} iterations)...',
            file=sys.stderr)
        records = self.parallel(
            joblib.delayed(self.solve_one)(models, problem_i, get_precedences) for problem_i in cases)

        problems_filtered = self.problems

        if self.isolated:
            for name in models:
                # We get the values only from the first iteration, assuming they don't differ across iterations.
                assert 'precedence_cost' not in df_data
                df_data[name, 'precedence_cost'] = [r.get((name, 'precedence_cost'), None) for r in
                                                    records[:len(problems_filtered)]]
                assert 'symbols' not in df_data
                df_data[name, 'symbols'] = [r.get((name, 'symbols'), None) for r in records[:len(problems_filtered)]]

        iter_dfs = []
        field_series = {k: [] for k in self.columns}
        for i in range(self.iterations):
            iter_records = records[i * len(problems_filtered):(i + 1) * len(problems_filtered)]
            iter_dicts = iter_records
            iter_df = dataframe_from_records(iter_dicts, dtypes=vampire.Result.pd_dtypes, index=problems_filtered)
            iter_dfs.append(iter_df)
            for k in self.columns:
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

        logs = self.evaluate_dataframe(main_df, 'symbol_cost', self.iterations, models.keys(), epoch)

        with self.tensorboard.train_writer.as_default():
            tf.summary.scalar('time/epoch/solver_eval', tf.timestamp() - time_begin, step=epoch)

        print(f'Solver evaluation after epoch {epoch}:\n{yaml.dump(logs)}')

        return logs

    def evaluate_dataframe(self, main_df, df_name, iterations, symbol_types, epoch=None):
        # If epoch is not specified, then TensorBoard logging is omitted.
        output_dir = os.path.join(self.name, df_name, f'epoch_{epoch}')

        save_df(main_df, os.path.join(output_dir, 'problems'))

        logs = {'problems/measured': len(main_df)}
        for dataset_name, dataset_problems in self.splits.items():
            df_dataset = main_df.loc[main_df.index.intersection(dataset_problems)]
            logs[dataset_name] = {'problems/split': len(dataset_problems), 'problems/measured&split': len(df_dataset)}
            dataset_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            with self.tensorboard.writer(dataset_name).as_default():
                for cat_name, cat_problems in self.problem_categories.items():
                    if dataset_name == 'train' and cat_name == 'all' and not self.train_without_questions:
                        continue
                    summary_name = f'{self.name}/{cat_name}'
                    if cat_problems is None:
                        records_df = df_dataset
                    else:
                        records_df = df_dataset.loc[df_dataset.index.intersection(cat_problems)]
                    records_df.index.to_series().to_csv(os.path.join(dataset_dir, f'{cat_name}.txt'),
                                                        header=False, index=False)

                    res = {'problems': {}}
                    if cat_problems is None:
                        res['problems']['category'] = len(self.problems)
                    else:
                        res['problems']['category'] = len(cat_problems)
                    res['problems']['measured&split&category'] = len(records_df)

                    df_success = records_df[[(i, 'returncode') for i in range(iterations)]] == 0
                    res['success'] = {
                        'count/mean': float(df_success.sum(axis=0).mean()),
                        'count/std': float(np.std(df_success.sum(axis=0))),
                        'rate/mean': float(df_success.mean(axis=0).mean())
                    }

                    # TODO: Compute maximum symmetric difference between iterations.
                    # TODO: Plot some distribution plots and scatter plots (e.g. 'precedence_cost' x success).

                    logs[dataset_name][cat_name] = res

                    if epoch is not None:
                        for k, v in self.flatten_logs(res).items():
                            tf.summary.scalar(f'{summary_name}/{k}', v, step=epoch)
                        if not self.baseline:
                            for symbol_type, column_name in itertools.product(symbol_types,
                                                                              ['symbols', 'precedence_cost']):
                                values = records_df[symbol_type, column_name]
                                tf.summary.histogram(f'{summary_name}/{symbol_type}/{column_name}', values, step=epoch)
                        for column_name in ['time_elapsed', 'saturation_iterations']:
                            values = records_df[[(i, column_name) for i in range(self.iterations)]].astype(float)
                            tf.summary.histogram(f'{summary_name}/{column_name}/all', values, step=epoch)

        with open(os.path.join(output_dir, 'logs.yaml'), 'w') as file:
            yaml.dump(logs, file)

        return logs

    def precedences(self, symbol_costs):
        precedences, precedence_costs = zip(*map(self.precedence, symbol_costs))
        precedence_costs = tf.stack(precedence_costs)
        return precedences, precedence_costs

    def precedence(self, symbol_cost):
        # We sort the symbols by cost in non-decreasing order.
        precedence = tf.argsort(symbol_cost, direction='DESCENDING')
        n_symbols = symbol_cost.shape[0]
        precedence_cost = tf.tensordot(tf.gather(symbol_cost, precedence), tf.range(n_symbols, dtype=symbol_cost.dtype),
                                       1) * 2 / (n_symbols * (n_symbols + 1))
        return precedence, precedence_cost

    @classmethod
    def solver_result_to_dict(cls, result):
        return {k: getattr(result, k, None) for k in cls.columns}

    def solve_one(self, models, problem_i, get_precedences=None):
        problem = self.problems[problem_i]
        record = {}
        with timer() as t_total:
            precedences = None
            if not self.baseline:
                if get_precedences is None:
                    precedences = {}
                    with timer() as t_preprocess:
                        for name, model in models.items():
                            # Attempt to predict a precedence
                            with timer() as t_predict:
                                assert self.isolated
                                res = model(tf.convert_to_tensor([problem]), training=False, cache=False)
                            record[name, 'time_predict'] = t_predict.elapsed
                            assert tf.shape(res['valid']) == (1,)
                            valid = res['valid'][0]
                            record[name, 'symbol_cost_valid'] = valid.numpy()
                            precedence = None
                            if valid:
                                # Prediction of the precedence was successful
                                assert res['costs'].nrows() == 1
                                symbol_cost = res['costs'][0]
                                record[name, 'symbols'] = tf.shape(symbol_cost)[0].numpy()
                                with timer() as t_sort:
                                    precedence, precedence_cost = self.precedence(symbol_cost)
                                    record[name, 'precedence_cost'] = precedence_cost.numpy()
                                record[name, 'time_sort'] = t_sort.elapsed
                            precedences[name] = precedence
                        if any(p is None for p in precedences.values()):
                            precedences = None
                    record['time_preprocess'] = t_preprocess.elapsed
                else:
                    precedences = get_precedences(problem_i)
            if self.baseline or precedences is not None:
                solver_res = self.solver.solve(problem, precedences=precedences, cache=self.cache)
                record.update(self.solver_result_to_dict(solver_res))
        record['time_total'] = t_total.elapsed
        return record
