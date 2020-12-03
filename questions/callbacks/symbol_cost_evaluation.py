import os

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from vampire_ml.results import save_df


class SymbolCostEvaluation(tf.keras.callbacks.Callback):
    def __init__(self, problems=None, start=0, step=1, output_dir=None):
        super().__init__()
        self.problems = problems
        if start is None and step is not None:
            start = 0
        elif step is None and start is not None:
            step = 1
        self.start = start
        self.step = step
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        if self.start is not None and self.step is not None and epoch >= self.start and (
                epoch - self.start) % self.step == 0:
            logs.update(self.evaluate(self.model.symbol_cost_model, epoch))
            print(f'Metrics after epoch {epoch}: {logs}')

    def evaluate(self, symbol_cost_model, epoch=None):
        logs = {}
        for dataset_name, dataset_problems in self.problems.items():
            if dataset_problems is not None and dataset_problems.cardinality() >= 1:
                print(f'Evaluating symbol cost model \'{symbol_cost_model.name}\' on {dataset_name} problems...')
                res = symbol_cost_model.evaluate(dataset_problems, return_dict=True)
                records_df = symbol_cost_model.solver_metric.result_df()
                logs.update({self.log_key(dataset_name, k): v for k, v in res.items() if k != 'loss'})
                if self.output_dir is not None:
                    output_dir = os.path.join(self.output_dir, 'solver_evaluation', symbol_cost_model.name,
                                              f'epoch_{epoch}', dataset_name)
                    save_df(records_df, os.path.join(output_dir, 'problems'))
                    for subset_name, subset_df in {'all': records_df,
                                                   'successful': records_df[records_df['returncode'] == 0]}.items():
                        subset_dir = os.path.join(output_dir, subset_name)
                        os.makedirs(subset_dir, exist_ok=True)
                        for column_name in ['time_elapsed', 'saturation_iterations']:
                            plt.figure()
                            sns.ecdfplot(subset_df, x=column_name)
                            plt.savefig(os.path.join(subset_dir, f'{column_name}.svg'))
                            plt.close()
                        for x, y in [('time_elapsed', 'saturation_iterations')]:
                            plt.figure()
                            sns.scatterplot(data=subset_df, x=x, y=y)
                            plt.savefig(os.path.join(subset_dir, f'{x}_vs_{y}.svg'))
                            plt.close()

        return logs

    @staticmethod
    def log_key(dataset_name, metric_name):
        assert dataset_name in {'train', 'validation'}
        if dataset_name == 'validation':
            return f'val_{metric_name}'
        return metric_name
