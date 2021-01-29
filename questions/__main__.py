#!/usr/bin/env python3

import datetime
import glob
import itertools
import logging
import os
import re
import socket
import sys
import warnings

import hydra
import joblib
import matplotlib.pyplot as plt
import neptune
import neptune_tensorboard
import neptunecontrib.monitoring.optuna
import numpy as np
import optuna
import pandas as pd
import scipy
import seaborn as sns
import tensorflow as tf
from attributedict.collections import AttributeDict
from omegaconf import DictConfig, OmegaConf
from ordered_set import OrderedSet

from proving import tptp
from proving.graphifier import Graphifier
from proving.graphifier import get_graphs
from proving.memory import memory
from proving.solver import Solver
from proving.utils import cardinality_finite
from proving.utils import dataframe_from_records
from proving.utils import py_str
from questions import callbacks
from questions import datasets
from questions import param
from questions import plot
from questions import trial
from questions.datasets.questions import Generator
from vampire_ml.results import save_df


def save_problems(problems, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.writelines(f'{py_str(p)}\n' for p in problems)
    neptune.log_artifact(filename)
    logging.info(f'List of problems saved: {filename}')


def flatten_config(cfg):
    return flatten_dict(OmegaConf.to_container(cfg))


def flatten_dict(d):
    return pd.json_normalize(d).to_dict(orient='records')[0]


@hydra.main(config_name='config')
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=cfg.log_level)
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    sys.setrecursionlimit(cfg.recursion_limit)

    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(cfg.tf.run_eagerly)
    tf.summary.experimental.set_step(0)

    # Neptune
    neptune.init(project_qualified_name=cfg.neptune.project_name)
    cfg_flat = flatten_config(cfg)
    neptune.create_experiment(params=cfg_flat, logger=logging.getLogger(),
                              **OmegaConf.to_container(cfg.neptune.experiment))
    if cfg.optuna.trials is None or cfg.optuna.trials <= 0:
        neptune_tensorboard.integrate_with_tensorflow(prefix=True)

    logging.info('Python recursion limit: %d', sys.getrecursionlimit())
    logging.info('TensorFlow inter-op parallelism threads: %d', tf.config.threading.get_inter_op_parallelism_threads())
    logging.info('TensorFlow intra-op parallelism threads: %d', tf.config.threading.get_intra_op_parallelism_threads())
    logging.info('TensorFlow physical devices: %s', tf.config.experimental.list_physical_devices())

    logging.info(f'Working directory: {os.getcwd()}')
    logging.info(f'Joblib cache location: {memory.location}')

    experiment_id = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    log_dir = os.path.join(hydra.utils.to_absolute_path(cfg.tb.logdir), experiment_id)
    logging.info(f'Log directory: {log_dir}')
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))

    with train_writer.as_default():
        # https://stackoverflow.com/a/61106106/4054250
        args_series = pd.Series(cfg_flat, name='value')
        args_series.index.name = 'parameter'
        tf.summary.text('config', args_series.to_markdown())
        tf.summary.text('command', ' '.join(sys.argv))
        logging.info('Command: %s', ' '.join(sys.argv))
        tf.summary.text('hostname', socket.gethostname())
        logging.info(f'Hostname: {socket.gethostname()}')

    def normalize_pattern(pattern):
        if re.match(
                r'^(?P<name>(?P<domain>[A-Z]{3})(?P<number>[0-9]{3})(?P<form>[-+^=_])(?P<version>[1-9])(?P<size_parameters>[0-9]*(\.[0-9]{3})*))$',
                pattern):
            # Pattern is a problem name without file extension.
            # Prepend the file extension '.p'.
            pattern = f'{pattern}.p'
        m = re.match(
            r'^(?P<name>(?P<domain>[A-Z]{3})(?P<number>[0-9]{3})(?P<form>[-+^=_])(?P<version>[1-9])(?P<size_parameters>[0-9]*(\.[0-9]{3})*))(?:\.[pg])$',
            pattern)
        if m:
            # Pattern is a problem base name without domain directory name.
            # Prepend domain directory name.
            pattern = os.path.join(m['domain'], pattern)
        return pattern

    patterns = list(map(normalize_pattern, cfg.problems.patterns))

    clausifier = Solver(options=OmegaConf.to_container(cfg.clausifier.options), timeout=cfg.clausifier.timeout)
    solver = Solver(options=OmegaConf.to_container(cfg.solver.options), timeout=cfg.solver.timeout)

    with joblib.parallel_backend('threading', n_jobs=cfg.jobs), joblib.Parallel(verbose=10) as parallel:
        # We need to split problems first and then collect questions for each of the datasets
        # because not all problems have questions and we only generate questions samples
        # for problems with at least one question.
        if cfg.problems.train is not None and cfg.problems.val is not None:
            logging.info('Loading problem splits from files...')
            problems = {
                'val': tf.data.TextLineDataset(hydra.utils.to_absolute_path(cfg.problems.val)),
                'train': tf.data.TextLineDataset(hydra.utils.to_absolute_path(cfg.problems.train))
            }
            problems_all = problems['val'].concatenate(problems['train'])
        else:
            logging.info('Collecting available problems...')
            problems_all = datasets.problems.get_dataset(patterns)
            save_problems(problems_all, os.path.join('problems', 'all.txt'))
            if cfg.problems.max_count is not None:
                problems_all = problems_all.take(cfg.problems.max_count)
            n_problems = cardinality_finite(problems_all)
            logging.info(f'Number of problems available: {n_problems}')
            assert 0 <= cfg.val_split <= 1
            problems_validation_count = tf.cast(tf.round(tf.cast(n_problems, tf.float32) * cfg.val_split), tf.int64)
            assert problems_validation_count >= 0
            problems = {
                'val': problems_all.take(problems_validation_count),
                'train': problems_all.skip(problems_validation_count)
            }
        save_problems(problems_all, os.path.join('problems', 'taken.txt'))
        logging.info('Number of problems taken: %d', cardinality_finite(problems_all))

        problem_records = {p: {**tptp.problem_properties(p), **{f'dataset_{k}': False for k in problems}} for p in
                           map(py_str, problems_all)}
        problem_records_types = {**tptp.property_types, **{f'dataset_{k}': np.bool for k in problems}}
        for k, p in problems.items():
            logging.info(f'Number of {k} problems: {cardinality_finite(p)}')
            save_problems(p, os.path.join('problems', 'dataset', f'{k}.txt'))
            for pp in map(py_str, p):
                problem_records[pp][f'dataset_{k}'] = True

        with train_writer.as_default():
            questions_dir = cfg.questions.dir
            if questions_dir is None:
                questions_dir = 'questions'
            else:
                questions_dir = hydra.utils.to_absolute_path(cfg.questions.dir)
            try:
                generator = Generator.load(questions_dir)
                logging.info('Generator loaded.')
                if any(l != r for l, r in itertools.zip_longest(generator.problems, map(py_str, problems_all))):
                    raise RuntimeError('Loaded generator uses different problems.')
                if set(generator.randomize) != set(cfg.questions.randomize):
                    raise RuntimeError(
                        f'Loaded generator randomizes different symbol type. Expected: {cfg.questions.randomize}. Actual: {generator.randomize}.')
                if generator.hoeffding_exponent != cfg.questions.hoeffding_exponent:
                    warnings.warn(
                        f'Loaded generator uses different Hoeffding exponent. Expected: {cfg.questions.hoeffding_exponent}. Actual: {generator.hoeffding_exponent}.')
            except FileNotFoundError:
                generator = Generator.fresh(list(map(py_str, problems_all)), clausifier,
                                            randomize=cfg.questions.randomize,
                                            hoeffding_exponent=cfg.questions.hoeffding_exponent)
                logging.info('Starting generating questions from scratch.')
            questions_all = generator.generate(solver,
                                               num_questions_per_batch=cfg.questions.batch_size,
                                               num_questions_per_problem=cfg.questions.max_per_problem,
                                               dir=questions_dir,
                                               num_questions=cfg.questions.max_count)

            question_counts = [q.shape[0] for q in questions_all.values()]
            signature_lengths = [q.shape[1] for q in questions_all.values()]

            try:
                print(f'Question counts: {scipy.stats.describe(question_counts)}')
            except ValueError:
                pass
            try:
                print(f'Signature sizes: {scipy.stats.describe(signature_lengths)}')
            except ValueError:
                pass

            df_index = pd.Index(questions_all.keys(), name='name')
            df = pd.DataFrame({
                'n_questions': pd.Series(question_counts, index=df_index, dtype=pd.UInt32Dtype(), name='n_questions'),
                'n_symbols': pd.Series(signature_lengths, index=df_index, dtype=pd.UInt32Dtype(), name='n_symbols')
            }, index=df_index)
            save_df(df, os.path.join('problems', 'with_questions'))

            figure = plt.figure(figsize=(8, 8))
            plt.title('Problems with questions')
            sns.scatterplot(x=signature_lengths, y=question_counts)
            plt.xlabel('Symbols')
            plt.ylabel('Questions')
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig(os.path.join('problems', 'with_questions.png'))
            image = plot.plot_to_image(figure)
            tf.summary.image('Problems with questions', image)

        for k, v in problem_records.items():
            if k in questions_all:
                v['num_questions'] = questions_all[k].shape[0]
                v['num_symbols'] = questions_all[k].shape[1]
            else:
                v['num_questions'] = 0
        problem_records_types.update({'num_questions': pd.UInt32Dtype(), 'num_symbols': pd.UInt32Dtype()})

        questions = {}
        question_batches = {}
        problems_to_graphify = OrderedSet()
        problems_with_questions = {}
        for k, p in problems.items():
            q = datasets.questions.individual.dict_to_dataset(questions_all, p).cache()
            problems_to_graphify.update(py_str(e['problem']) for e in q)
            questions[k] = q
            question_batches[k] = datasets.questions.batch.batch(q, cfg.batch_size[k]).cache()
            problems_with_questions[k] = [pp for pp in map(py_str, p) if pp in questions_all]
            logging.info(f'Number of {k} problems with questions: {len(problems_with_questions[k])}')

        checkpoint_dir = 'tf_ckpts'
        epoch_ckpt_dir = os.path.join(checkpoint_dir, 'epoch')
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        for f in glob.iglob(os.path.join(epoch_ckpt_dir, 'weights.*.tf.*')):
            os.remove(f)
        acc_ckpt_dir = os.path.join(checkpoint_dir, 'val_binary_accuracy')
        os.makedirs(acc_ckpt_dir, exist_ok=True)
        for f in glob.iglob(os.path.join(acc_ckpt_dir, 'weights.*.tf.*')):
            os.remove(f)
        success_ckpt_dir = os.path.join(checkpoint_dir, 'val_solver_success_rate')
        os.makedirs(success_ckpt_dir, exist_ok=True)
        for f in glob.iglob(os.path.join(success_ckpt_dir, 'weights.*.tf.*')):
            os.remove(f)
        tensorboard = callbacks.TensorBoard(log_dir=log_dir, profile_batch=cfg.tb.profile_batch, histogram_freq=1,
                                            embeddings_freq=1)
        cbs = [
            tensorboard,
            callbacks.Time(problems={k: next(iter(v.take(32).batch(32))) for k, v in problems.items() if
                                     cardinality_finite(v) > 0},
                           tensorboard=tensorboard),
            tf.keras.callbacks.CSVLogger('epochs.csv'),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(epoch_ckpt_dir, 'weights.{epoch:05d}-{val_binary_accuracy:.2f}.tf'),
                save_weights_only=True, verbose=0),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(acc_ckpt_dir, 'weights.{epoch:05d}-{val_binary_accuracy:.2f}.tf'),
                save_weights_only=True, verbose=1, monitor='val_binary_accuracy', save_best_only=True),
            tf.keras.callbacks.EarlyStopping(**cfg.early_stopping)
        ]

        symbol_cost_evaluation_callback = None
        if cfg.solver_eval.enable:
            solver_eval_problems = problems['val']
            if cfg.solver_eval.problems.val is not None and cfg.solver_eval.problems.val >= 0:
                solver_eval_problems = solver_eval_problems.take(cfg.solver_eval.problems.val)
            if cfg.solver_eval.train_without_questions:
                solver_eval_problems_train = problems['train']
            else:
                solver_eval_problems_train = tf.data.Dataset.from_tensor_slices(problems_with_questions['train'])
            if cfg.solver_eval.problems.train is not None and cfg.solver_eval.problems.train >= 0:
                solver_eval_problems_train = solver_eval_problems_train.take(cfg.solver_eval.problems.train)
            if cardinality_finite(solver_eval_problems_train, 1) >= 1:
                solver_eval_problems = solver_eval_problems.concatenate(solver_eval_problems_train)
            solver_eval_problems = list(OrderedSet(map(py_str, solver_eval_problems)))
            problems_to_graphify.update(solver_eval_problems)

            problem_categories = {'all': None, 'with_questions': questions_all.keys()}
            for cat_name, cat_filename in cfg.problems.sets:
                with open(cat_filename) as f:
                    problem_categories[cat_name] = [l.rstrip('\n') for l in f]

            symbol_cost_evaluation_callback = callbacks.SymbolCostEvaluation(
                'epochs_solver_eval.csv',
                solver=solver,
                cache=cfg.solver_eval.cache,
                problems=solver_eval_problems,
                symbol_type=cfg.symbol_type,
                splits={k: list(map(py_str, v)) for k, v in problems.items()},
                batch_size=cfg.solver_eval.batch_size,
                start=cfg.solver_eval.start,
                step=cfg.solver_eval.step,
                iterations=cfg.solver_eval.iterations,
                tensorboard=tensorboard,
                problem_categories=problem_categories,
                baseline=cfg.symbol_cost.model == 'baseline',
                parallel=parallel,
                train_without_questions=cfg.solver_eval.train_without_questions)
            cbs.append(symbol_cost_evaluation_callback)

        state = AttributeDict({
            'symbol_cost_evaluation_callback': symbol_cost_evaluation_callback,
            'tensorboard': tensorboard,
            'question_batches': question_batches,
            'questions': questions,
            'cbs': cbs,
            'clausifier': clausifier
        })

        if cfg.symbol_embedding_model == 'gcn':
            graphifier = Graphifier(clausifier, max_number_of_nodes=cfg.gcn.max_problem_nodes)
            # problems_to_graphify = set(map(py_str, problems_all))
            graphs, graphs_df = get_graphs(graphifier, problems_to_graphify)
            for problem_name, rec in graphs_df.iterrows():
                problem_records[problem_name].update(rec.to_dict())
            logging.info(f'Number of problems graphified: {len(graphs)}')
            save_df(graphs_df, 'graphs')
            state.update({
                'graphifier': graphifier,
                'graphs': graphs
            })

        save_df(dataframe_from_records(list(problem_records.values()), index_keys='name', dtypes=problem_records_types),
                'problems')

        if cfg.optuna.trials is None or cfg.optuna.trials <= 0:
            trial.run(cfg, state)
        else:
            if cfg.load_checkpoint is not None:
                raise RuntimeError('Cannot combine loading checkpoint with Optuna hyperparameter optimization.')

            def objective(t):
                params = param.suggest(t, trial.space)
                cfg_merged = OmegaConf.merge(cfg, params)
                return trial.run(cfg_merged, state, monitor=cfg.optuna.monitor, optuna_trial=t)

            study = optuna.create_study()
            neptune_callback = neptunecontrib.monitoring.optuna.NeptuneCallback(log_study=True)
            study.optimize(objective, n_trials=cfg.optuna.trials, callbacks=[neptune_callback])
            print(study.best_params)


if __name__ == '__main__':
    main()
