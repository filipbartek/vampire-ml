#!/usr/bin/env python3

import glob
import itertools
import logging
import os
import re
import socket
import sys

import hydra
import joblib
import matplotlib.pyplot as plt
import neptune
import neptune_tensorboard
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import tensorflow as tf
import yaml
from omegaconf import DictConfig, OmegaConf
from ordered_set import OrderedSet
from tqdm import tqdm

from proving import tptp
from proving.graphifier import Graphifier
from proving.graphifier import get_graphs
from proving.memory import memory
from proving.solver import Solver
from proving.utils import cardinality_finite
from proving.utils import dataframe_from_records
from proving.utils import flatten_dict
from proving.utils import py_str
from questions import callbacks
from questions import datasets
from questions import models
from questions import plot
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


@hydra.main(config_name='config')
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=cfg.log_level)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(cfg.tf.run_eagerly)
    tf.summary.experimental.set_step(0)
    if cfg.recursion_limit is not None:
        sys.setrecursionlimit(cfg.recursion_limit)

    # Neptune
    neptune.init(project_qualified_name=cfg.neptune.project_qualified_name)
    neptune.create_experiment(params=flatten_config(cfg), logger=logging.getLogger(),
                              upload_source_files=map(hydra.utils.to_absolute_path,
                                                      cfg.neptune.experiment.upload_source_files),
                              **{k: v for k, v in OmegaConf.to_container(cfg.neptune.experiment).items() if
                                 k != 'upload_source_files'})
    neptune_tensorboard.integrate_with_tensorflow(prefix=True)

    logging.info(f'Working directory: {os.getcwd()}')
    neptune.set_property('cwd', os.getcwd())
    neptune.set_property('original_cwd', hydra.utils.get_original_cwd())
    neptune.set_property('cwd_relpath', os.path.relpath(os.getcwd(), hydra.utils.get_original_cwd()))

    logging.info('Python recursion limit: %d', sys.getrecursionlimit())
    neptune.set_property('recursion_limit', sys.getrecursionlimit())
    logging.info('TensorFlow inter-op parallelism threads: %d', tf.config.threading.get_inter_op_parallelism_threads())
    logging.info('TensorFlow intra-op parallelism threads: %d', tf.config.threading.get_intra_op_parallelism_threads())
    logging.info('TensorFlow physical devices: %s', tf.config.experimental.list_physical_devices())
    neptune.set_property('tf.physical_devices', tf.config.experimental.list_physical_devices())

    logging.info(f'Joblib cache location: {memory.location}')
    neptune.set_property('joblib.cache.location', memory.location)

    writer_train = tf.summary.create_file_writer('train')
    with writer_train.as_default():
        # https://stackoverflow.com/a/61106106/4054250
        args_series = pd.Series(cfg.__dict__, name='value')
        args_series.index.name = 'argument'
        tf.summary.text('args', args_series.to_markdown())
        tf.summary.text('command', ' '.join(sys.argv))
        logging.info('Command: %s', ' '.join(sys.argv))
        neptune.set_property('command', ' '.join(sys.argv))
        tf.summary.text('hostname', socket.gethostname())
        logging.info(f'Hostname: {socket.gethostname()}')

    patterns = cfg.problems.patterns

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

    patterns = list(map(normalize_pattern, patterns))

    clausifier = Solver(**OmegaConf.to_container(cfg.clausifier))
    solver = Solver(**OmegaConf.to_container(cfg.solver))

    with joblib.parallel_backend('threading', n_jobs=cfg.jobs), joblib.Parallel(verbose=10) as parallel:
        # We need to split problems first and then collect questions for each of the datasets
        # because not all problems have questions and we only generate questions samples
        # for problems with at least one question.
        if cfg.problems.train is not None and cfg.problems.val is not None:
            problems = {
                'val': tf.data.TextLineDataset(hydra.utils.to_absolute_path(cfg.problems.val)),
                'train': tf.data.TextLineDataset(hydra.utils.to_absolute_path(cfg.problems.train))
            }
            problems_all = problems['val'].concatenate(problems['train'])
        else:
            logging.info('Collecting available problems...')
            if cfg.problems.names is None:
                problems_all = datasets.problems.get_dataset(patterns)
            else:
                problems_all = tf.data.TextLineDataset(cfg.problems.names)
            save_problems(problems_all, os.path.join('problems', 'all.txt'))
            if cfg.problems.max_count is not None:
                problems_all = problems_all.take(cfg.problems.max_count)
            save_problems(problems_all, os.path.join('problems', 'taken.txt'))
            n_problems = cardinality_finite(problems_all)
            logging.info('Number of problems available: %d', n_problems)
            assert 0 <= cfg.validation_split <= 1
            problems_validation_count = tf.cast(tf.round(tf.cast(n_problems, tf.float32) * cfg.validation_split),
                                                tf.int64)
            assert problems_validation_count >= 0
            problems = {
                'val': problems_all.take(problems_validation_count),
                'train': problems_all.skip(problems_validation_count)
            }
        logging.info('Number of problems taken: %d', cardinality_finite(problems_all))
        neptune.set_property('problems/taken', cardinality_finite(problems_all))

        problem_records = {p: {**tptp.problem_properties(p), **{f'dataset_{k}': False for k in problems}} for p in
                           map(py_str, problems_all)}
        problem_records_types = {**tptp.property_types, **{f'dataset_{k}': np.bool for k in problems}}
        for k, p in problems.items():
            logging.info(f'Number of {k} problems: {cardinality_finite(p)}')
            neptune.set_property(f'problems/taken/{k}', cardinality_finite(p))
            save_problems(p, os.path.join('problems', 'dataset', f'{k}.txt'))
            for pp in map(py_str, p):
                problem_records[pp][f'dataset_{k}'] = True

        with writer_train.as_default():
            if cfg.questions.dir_legacy is None:
                questions_dir = cfg.questions.dir
                if questions_dir is None:
                    questions_dir = 'questions'
                else:
                    questions_dir = hydra.utils.to_absolute_path(questions_dir)
                try:
                    generator = Generator.load(questions_dir)
                    logging.info('Generator loaded.')
                    if any(l != r for l, r in itertools.zip_longest(generator.problems, map(py_str, problems_all))):
                        raise RuntimeError('Loaded generator uses different problems.')
                    if set(generator.randomize) != set(cfg.questions.randomize):
                        raise RuntimeError(
                            f'Loaded generator randomizes different symbol type. Expected: {cfg.questions.randomize}. Actual: {generator.randomize}.')
                except FileNotFoundError:
                    generator = Generator.fresh(list(map(py_str, problems_all)), clausifier,
                                                randomize=cfg.questions.randomize,
                                                hoeffding_exponent=cfg.questions.hoeffding_exponent)
                    logging.info('Starting generating questions from scratch.')
                with writer_train.as_default():
                    questions_all = generator.generate(solver,
                                                       num_questions_per_batch=cfg.questions.batch_size,
                                                       num_questions_per_problem=cfg.questions.max_per_problem,
                                                       dir=questions_dir,
                                                       num_questions=cfg.questions.max_count)
            else:
                # TODO?: Only load questions if the batches are not cached.
                questions_file = os.path.join(hydra.utils.to_absolute_path('cache'),
                                              f'symbol_type_{cfg.symbol_type}',
                                              f'max_questions_per_problem_{cfg.questions.max_per_problem}',
                                              'questions.pkl')

                # Here we load the raw, un-normalized questions (oriented element-wise differences of inverse precedences).
                questions_all = datasets.questions.load_questions.load(questions_file, cfg.questions.dir_legacy,
                                                                       cfg.questions.max_per_problem)

            neptune.set_property('problems/with_questions', len(questions_all))

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
            q = datasets.questions.individual.dict_to_dataset(questions_all, p, normalize=cfg.questions.normalize).cache()
            problems_to_graphify.update(py_str(e['problem']) for e in q)
            questions[k] = q
            batch_size = {'train': cfg.batch_size.train, 'val': cfg.batch_size.val}[k]
            question_batches[k] = datasets.questions.batch.batch(q, batch_size).cache()
            problems_with_questions[k] = [pp for pp in map(py_str, p) if pp in questions_all]
            logging.info(f'Number of {k} problems with questions: {len(problems_with_questions[k])}')
            neptune.set_property(f'problems/with_questions/{k}', len(problems_with_questions[k]))

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
        tensorboard = callbacks.TensorBoard(log_dir='.', profile_batch=cfg.tb.profile_batch, histogram_freq=1,
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
            tf.keras.callbacks.EarlyStopping(**cfg.early_stopping),
            tf.keras.callbacks.ReduceLROnPlateau(**cfg.reduce_lr_on_plateau)
        ]

        solver_eval_problems = None
        if cfg.solver_eval.start is not None or cfg.solver_eval.step is not None:
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

        logging.info(f'Symbol cost model: {cfg.symbol_cost.model}')
        if cfg.symbol_cost.model == 'baseline':
            model_symbol_cost = models.symbol_cost.Baseline()
        else:
            if cfg.symbol_cost.model == 'direct':
                model_symbol_cost = models.symbol_cost.Direct(questions_all)
            elif cfg.symbol_cost.model == 'composite':
                embedding_to_cost = None
                logging.info(f'Symbol embedding model: {cfg.symbol_embedding_model}')
                if cfg.symbol_embedding_model == 'simple':
                    model_symbol_embedding = models.symbol_features.Simple(clausifier, cfg.symbol_type)
                    if cfg.embedding_to_cost.hidden.units > 0:
                        cbs.append(callbacks.Weights(tensorboard))
                    if cfg.simple_model_kernel is not None:
                        kernel = np.fromstring(cfg.simple_model_kernel, count=model_symbol_embedding.n, sep=',')
                        logging.info(f'Simple model kernel: {kernel}')
                        embedding_to_cost = tf.keras.layers.Dense(1, use_bias=False, trainable=False,
                                                                  kernel_initializer=tf.constant_initializer(kernel))

                elif cfg.symbol_embedding_model == 'gcn':
                    graphifier = Graphifier(clausifier, max_number_of_nodes=cfg.gcn.max_problem_nodes)
                    # problems_to_graphify = set(map(py_str, problems_all))
                    graphs, graphs_df = get_graphs(graphifier, problems_to_graphify)
                    for problem_name, rec in graphs_df.iterrows():
                        problem_records[problem_name].update(rec.to_dict())
                    logging.info(f'Number of problems graphified: {len(graphs)}')
                    neptune.set_property('problems/graphified', len(graphs))
                    save_df(graphs_df, 'graphs')

                    gcn = models.symbol_features.GCN(cfg.gcn, graphifier.canonical_etypes, graphifier.ntype_in_degrees,
                                                     graphifier.ntype_feat_sizes, output_ntypes=[cfg.symbol_type])
                    model_symbol_embedding = models.symbol_features.Graph(graphifier, graphs, cfg.symbol_type, gcn)
                else:
                    raise ValueError(f'Unsupported symbol embedding model: {cfg.symbol_embedding_model}')
                if embedding_to_cost is None:
                    if cfg.embedding_to_cost.hidden.units <= 0:
                        embedding_to_cost = tf.keras.layers.Dense(1, name='embedding_to_cost',
                                                                  kernel_regularizer=tf.keras.regularizers.L1L2(
                                                                      l1=cfg.embedding_to_cost.l1,
                                                                      l2=cfg.embedding_to_cost.l2))
                    else:
                        embedding_to_cost = tf.keras.Sequential([
                            tf.keras.layers.Dense(cfg.embedding_to_cost.hidden.units,
                                                  activation=cfg.embedding_to_cost.hidden.activation,
                                                  kernel_regularizer=tf.keras.regularizers.L1L2(
                                                      l1=cfg.embedding_to_cost.l1,
                                                      l2=cfg.embedding_to_cost.l2)),
                            tf.keras.layers.Dense(1,
                                                  kernel_regularizer=tf.keras.regularizers.L1L2(
                                                      l1=cfg.embedding_to_cost.l1,
                                                      l2=cfg.embedding_to_cost.l2))
                        ], name='embedding_to_cost')
                model_symbol_cost = models.symbol_cost.Composite(model_symbol_embedding, embedding_to_cost,
                                                                 l2=cfg.symbol_cost.l2)
            else:
                raise ValueError(f'Unsupported symbol cost model: {cfg.symbol_cost.model}')

        save_df(dataframe_from_records(list(problem_records.values()), index_keys='name', dtypes=problem_records_types),
                'problems')

        model_logit = models.question_logit.QuestionLogitModel(model_symbol_cost)

        # We need to set_model before we begin using tensorboard. Tensorboard is used in other callbacks in symbol cost evaluation.
        tensorboard.set_model(model_logit)

        if solver_eval_problems is not None:
            problem_categories = {
                'all': None,
                'with_questions': questions_all.keys(),
                'graphified': graphs.keys(),
                'with_questions&graphified': OrderedSet(questions_all.keys()) & graphs.keys()
            }
            for cat_name, cat_filename in cfg.solver_eval.problem_set:
                with open(cat_filename) as f:
                    problem_categories[cat_name] = [l.rstrip('\n') for l in f]

            symbol_cost_evaluation_callback = callbacks.SymbolCostEvaluation(
                cfg.solver_eval,
                'epochs_solver_eval.csv',
                solver=solver,
                problems=solver_eval_problems,
                symbol_type=cfg.symbol_type,
                splits={k: list(map(py_str, v)) for k, v in problems.items()},
                tensorboard=tensorboard,
                problem_categories=problem_categories,
                baseline=cfg.symbol_cost.model == 'baseline',
                parallel=parallel)
            cbs.append(symbol_cost_evaluation_callback)

            for name, d in cfg.solver_eval.baselines.items():
                df = pd.read_pickle(hydra.utils.to_absolute_path(d.filename))
                logs = symbol_cost_evaluation_callback.evaluate_dataframe(df, name, d.iterations)
                print(f'Baseline \'{name}\':\n{yaml.dump(logs)}')

            if symbol_cost_evaluation_callback.start <= -1:
                symbol_cost_evaluation_callback.evaluate(symbol_cost_model=model_symbol_cost, epoch=-1)

        if not isinstance(model_symbol_cost, models.symbol_cost.Baseline):
            Optimizer = {
                'sgd': tf.keras.optimizers.SGD,
                'adam': tf.keras.optimizers.Adam,
                'rmsprop': tf.keras.optimizers.RMSprop
            }[cfg.optimizer]
            # https://arxiv.org/pdf/1706.02677.pdf
            # https://arxiv.org/abs/1711.00489
            learning_rate = cfg.learning_rate * cfg.batch_size.train
            neptune.set_property('learning_rate_scaled', learning_rate)
            optimizer = Optimizer(learning_rate=learning_rate)

            model_logit.compile(optimizer=optimizer)

            if cfg.restore_checkpoint is not None:
                model_logit.load_weights(hydra.utils.to_absolute_path(cfg.restore_checkpoint))
                logging.info(f'Checkpoint restored: {hydra.utils.to_absolute_path(cfg.restore_checkpoint)}')

            if not cfg.initial_eval:
                print('Initial evaluation of question logit model...')
                for k in question_batches:
                    print(f'Evaluating logit model on {k} questions...')
                    if k == 'train':
                        x = datasets.questions.batch.batch(questions[k], cfg.batch_size.val)
                    else:
                        x = question_batches[k]
                    metrics = model_logit.evaluate(x)
                    print(metrics)

            if cfg.initial_evaluation_extra:
                initial_evaluation(model_logit, questions_all, problems_all, cfg.batch_size.train)

            if cfg.epochs >= 1:
                print('Training...')
                model_logit.fit(question_batches['train'], validation_data=question_batches['val'], epochs=cfg.epochs,
                                callbacks=cbs)


def initial_evaluation(model_logit, questions_all, problems_all, batch_size, print_each_problem=False):
    print('Initial evaluation...')
    ds_individual = datasets.questions.individual.dict_to_dataset(questions_all, problems_all)
    print(f'Evaluating with batch size {batch_size}...')
    eval_res = model_logit.evaluate(datasets.questions.batch.batch(ds_individual, batch_size), return_dict=True)
    print(f'Evaluation result with batch size {batch_size}: {eval_res}')
    ds_batches = datasets.questions.batch.batch(ds_individual, 1)
    print('Evaluating with batch size 1...')
    eval_res = model_logit.evaluate(ds_batches, return_dict=True)
    print(f'Evaluation result with batch size 1: {eval_res}')
    batch_sizes = []
    batch_losses = []
    logit_lists = []
    for batch in tqdm(ds_batches, disable=print_each_problem):
        problem_names = [py_str(p) for p in batch['problems']]
        eval_res = model_logit.evaluate(tf.data.Dataset.from_tensors(batch), return_dict=True, verbose=0)
        call_res = model_logit(batch, training=False)
        if print_each_problem:
            print(f'{problem_names}: {eval_res}, {call_res}')
        batch_sizes.append(len(problem_names))
        batch_losses.append(eval_res['loss'])
        logit_lists.append(call_res.flat_values)
    print('Weighted average loss (expected overall loss): ', np.average(batch_losses, weights=batch_sizes))
    print('Mean of batch losses: ', np.mean(batch_losses))
    print('Median of batch losses (expected: 0.69): ', np.median(batch_losses))
    logits = tf.concat(logit_lists, axis=0)
    print('Mean of question logits (expected: 0): ', np.mean(logits))
    print('Mean of question accuracies (expected: 0.5): ', np.mean(logits.numpy() > 0))
    accs = []
    for logit in logit_lists:
        n_correct = np.count_nonzero(logit.numpy() > 0)
        acc = n_correct / len(logit)
        accs.append(acc)
    print('Weighted average accuracy (expected overall accuracy): ', np.mean(accs))


if __name__ == '__main__':
    main()
