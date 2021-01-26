#!/usr/bin/env python3

import argparse
import datetime
import glob
import itertools
import logging
import os
import re
import socket
import sys

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
from ordered_set import OrderedSet
from tqdm import tqdm

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name')
    parser.add_argument('problem', nargs='*')
    parser.add_argument('--problems', action='append')
    parser.add_argument('--problems-validation', action='append')
    parser.add_argument('--problems-train', action='append')
    parser.add_argument('--questions-dir')
    parser.add_argument('--questions-dir-legacy')
    parser.add_argument('--max-questions-per-problem', type=int)
    parser.add_argument('--max-problems', type=int, default=None)
    parser.add_argument('--logs-dir', default='logs')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--log-level', choices=['INFO', 'DEBUG'], default='INFO')
    parser.add_argument('--validation-split', type=float, default=0.5)
    parser.add_argument('--train-batch-size', type=int, default=32)
    parser.add_argument('--val-batch-size', type=int, default=256)
    parser.add_argument('--symbol-type', choices=['predicate', 'function'], default='predicate')
    parser.add_argument('--solver-eval-start', type=int, default=None,
                        help='Set to -1 to evaluate before first training epoch.')
    parser.add_argument('--solver-eval-step', type=int, default=None)
    parser.add_argument('--solver-eval-iterations', type=int, default=1)
    parser.add_argument('--solver-eval-batch-size', type=int, default=1000)
    parser.add_argument('--solver-eval-train-problems', type=int, default=1000)
    parser.add_argument('--solver-eval-val-problems', type=int, default=1000)
    parser.add_argument('--solver-eval-train-without-questions', action='store_true')
    parser.add_argument('--problem-set', action='append', nargs=2, default=[])
    parser.add_argument('--profile-batch', default=0)
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--run-eagerly', action='store_true')
    parser.add_argument('--symbol-embedding-model', default='gcn', choices=['simple', 'gcn'])
    parser.add_argument('--symbol-cost-model', default='composite', choices=['composite', 'direct', 'baseline'])
    parser.add_argument('--symbol-cost-l2', type=float, default=0,
                        help='Factor of L2 regularization penalty on symbol cost values')
    parser.add_argument('--embedding-to-cost-l1', type=float, default=0)
    parser.add_argument('--embedding-to-cost-l2', type=float, default=0)
    parser.add_argument('--embedding-to-cost-hidden-layer', type=int)
    parser.add_argument('--simple-model-kernel')
    parser.add_argument('--output')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--max-num-nodes', type=int, default=100000)
    parser.add_argument('--initial-evaluation-extra', action='store_true')
    # Python default is 1000. 10000 is enough to parse all TPTP problems.
    parser.add_argument('--recursion-limit', type=int, default=10000)
    parser.add_argument('--restore-checkpoint')
    parser.add_argument('--gcn-depth', type=int, default=4)
    parser.add_argument('--gcn-message-size', type=int, default=64)
    parser.add_argument('--gcn-activation', default='relu', choices=['relu', 'sigmoid'])
    parser.add_argument('--gcn-dropout', type=float)
    parser.add_argument('--no-layer-norm', action='store_true')
    parser.add_argument('--no-residual', action='store_true')
    parser.add_argument('--conv-norm', default='both', choices=['both', 'right', 'none'])
    parser.add_argument('--questions', type=int)
    parser.add_argument('--questions-per-batch', type=int, default=1000)
    parser.add_argument('--questions-per-problem', type=int)
    parser.add_argument('--questions-randomize', nargs='+')
    parser.add_argument('--hoeffding-exponent', type=float, default=4)
    parser.add_argument('--solver-options', type=yaml.safe_load, default={},
                        help='Options passed to Vampire. '
                             'Run `vampire --show_options on --show_experimental_options on` to print the options '
                             'supported by Vampire. '
                             'Format: YAML dictionary. '
                             'For example, "{time_limit: 10}" translates into '
                             '"--time_limit 10".'
                             'Recommended options: include, time_limit.')
    parser.add_argument('--clausifier-options', type=yaml.safe_load, default={})
    parser.add_argument('--solver-timeout', type=float, default=20,
                        help='Time in seconds after which each Vampire call is terminated.')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(args.run_eagerly)
    tf.summary.experimental.set_step(0)
    if args.recursion_limit is not None:
        sys.setrecursionlimit(args.recursion_limit)

    # Neptune
    neptune.init(project_qualified_name='filipbartek/vampire-ml')
    neptune.create_experiment(name=args.experiment_name, params=args.__dict__,
                              upload_source_files=['requirements.txt', 'questions/**/*.py', 'proving/**/*.py',
                                                   'vampire_ml/**/*.py'],
                              logger=logging.getLogger())
    neptune_tensorboard.integrate_with_tensorflow(prefix=True)

    experiment_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.experiment_name is not None:
        experiment_id = os.path.join(experiment_id, args.experiment_name)

    output = args.output
    if output is None:
        output = os.path.join('out', experiment_id)
    logging.info(f'Output directory: {output}')

    logging.info('Python recursion limit: %d', sys.getrecursionlimit())
    logging.info('TensorFlow inter-op parallelism threads: %d', tf.config.threading.get_inter_op_parallelism_threads())
    logging.info('TensorFlow intra-op parallelism threads: %d', tf.config.threading.get_intra_op_parallelism_threads())
    logging.info('TensorFlow physical devices: %s', tf.config.experimental.list_physical_devices())

    logging.info(f'Joblib cache location: {memory.location}')

    log_dir = os.path.join(args.logs_dir, experiment_id)
    logging.info(f'Log directory: {log_dir}')
    writer_train = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    with writer_train.as_default():
        # https://stackoverflow.com/a/61106106/4054250
        args_series = pd.Series(args.__dict__, name='value')
        args_series.index.name = 'argument'
        tf.summary.text('args', args_series.to_markdown())
        tf.summary.text('command', ' '.join(sys.argv))
        logging.info('Command: %s', ' '.join(sys.argv))
        tf.summary.text('hostname', socket.gethostname())
        logging.info(f'Hostname: {socket.gethostname()}')

    patterns = args.problem
    if patterns is None or len(patterns) == 0:
        patterns = ['**/*-*.p', '**/*+*.p']
        logging.info('Defaulting problem patterns to: %s', patterns)

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

    default_options = {'encode': 'on'}

    clausifier_options = {**default_options, 'time_limit': '300'}
    clausifier_options.update(args.clausifier_options)
    clausifier = Solver(options=clausifier_options)

    solver_options = {
        **default_options,
        'statistics': 'full',
        'time_statistics': 'on',
        'proof': 'off',
        'avatar': 'off',
        'saturation_algorithm': 'discount',
        'age_weight_ratio': '10',
        'literal_comparison_mode': 'predicate',
        'symbol_precedence': 'frequency',
        'time_limit': '10'
    }
    solver_options.update(args.solver_options)
    solver = Solver(options=solver_options, timeout=args.solver_timeout)

    with joblib.parallel_backend('threading', n_jobs=args.jobs), joblib.Parallel(verbose=10) as parallel:
        # We need to split problems first and then collect questions for each of the datasets
        # because not all problems have questions and we only generate questions samples
        # for problems with at least one question.
        if args.problems_train is not None and args.problems_validation is not None:
            problems = {
                'val': tf.data.TextLineDataset(args.problems_validation),
                'train': tf.data.TextLineDataset(args.problems_train)
            }
            problems_all = problems['val'].concatenate(problems['train'])
        else:
            logging.info('Collecting available problems...')
            if args.problems is None:
                problems_all = datasets.problems.get_dataset(patterns)
            else:
                problems_all = tf.data.TextLineDataset(args.problems)
            save_problems(problems_all, os.path.join(output, 'problems', 'all.txt'))
            if args.max_problems is not None:
                problems_all = problems_all.take(args.max_problems)
            save_problems(problems_all, os.path.join(output, 'problems', 'taken.txt'))
            n_problems = cardinality_finite(problems_all)
            logging.info('Number of problems available: %d', n_problems)
            assert 0 <= args.validation_split <= 1
            problems_validation_count = tf.cast(tf.round(tf.cast(n_problems, tf.float32) * args.validation_split),
                                                tf.int64)
            assert problems_validation_count >= 0
            problems = {
                'val': problems_all.take(problems_validation_count),
                'train': problems_all.skip(problems_validation_count)
            }
        logging.info('Number of problems taken: %d', cardinality_finite(problems_all))

        problem_records = {p: {**tptp.problem_properties(p), **{f'dataset_{k}': False for k in problems}} for p in
                           map(py_str, problems_all)}
        problem_records_types = {**tptp.property_types, **{f'dataset_{k}': np.bool for k in problems}}
        for k, p in problems.items():
            logging.info(f'Number of {k} problems: {cardinality_finite(p)}')
            save_problems(p, os.path.join(output, 'problems', 'dataset', f'{k}.txt'))
            for pp in map(py_str, p):
                problem_records[pp][f'dataset_{k}'] = True

        with writer_train.as_default():
            if args.questions_dir_legacy is None:
                questions_dir = args.questions_dir
                if questions_dir is None:
                    questions_dir = os.path.join(output, 'questions')
                try:
                    generator = Generator.load(questions_dir)
                    logging.info('Generator loaded.')
                    if any(l != r for l, r in itertools.zip_longest(generator.problems, map(py_str, problems_all))):
                        raise RuntimeError('Loaded generator uses different problems.')
                    if set(generator.randomize) != set(args.questions_randomize):
                        raise RuntimeError(
                            f'Loaded generator randomizes different symbol type. Expected: {args.questions_randomize}. Actual: {generator.randomize}.')
                except FileNotFoundError:
                    generator = Generator.fresh(list(map(py_str, problems_all)), clausifier,
                                                randomize=args.questions_randomize,
                                                hoeffding_exponent=args.hoeffding_exponent)
                    logging.info('Starting generating questions from scratch.')
                with writer_train.as_default():
                    questions_all = generator.generate(solver,
                                                       num_questions_per_batch=args.questions_per_batch,
                                                       num_questions_per_problem=args.questions_per_problem,
                                                       dir=questions_dir,
                                                       num_questions=args.questions)
            else:
                # TODO?: Only load questions if the batches are not cached.
                questions_file = os.path.join('cache',
                                              f'symbol_type_{args.symbol_type}',
                                              f'max_questions_per_problem_{args.max_questions_per_problem}',
                                              'questions.pkl')

                # Here we load the raw, un-normalized questions (oriented element-wise differences of inverse precedences).
                questions_all = datasets.questions.load_questions.load(questions_file, args.questions_dir_legacy,
                                                                       args.max_questions_per_problem)

            question_counts = [q.shape[0] for q in questions_all.values()]
            signature_lengths = [q.shape[1] for q in questions_all.values()]

            print(f'Question counts: {scipy.stats.describe(question_counts)}')
            print(f'Signature sizes: {scipy.stats.describe(signature_lengths)}')

            df_index = pd.Index(questions_all.keys(), name='name')
            df = pd.DataFrame({
                'n_questions': pd.Series(question_counts, index=df_index, dtype=pd.UInt32Dtype(), name='n_questions'),
                'n_symbols': pd.Series(signature_lengths, index=df_index, dtype=pd.UInt32Dtype(), name='n_symbols')
            }, index=df_index)
            save_df(df, os.path.join(output, 'problems', 'with_questions'))

            figure = plt.figure(figsize=(8, 8))
            plt.title('Problems with questions')
            sns.scatterplot(x=signature_lengths, y=question_counts)
            plt.xlabel('Symbols')
            plt.ylabel('Questions')
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig(os.path.join(output, 'problems', 'with_questions.png'))
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
        problems_to_graphify = OrderedSet()
        problems_with_questions = {}
        for k, p in problems.items():
            q = datasets.questions.individual.dict_to_dataset(questions_all, p)
            problems_to_graphify.update(py_str(e['problem']) for e in q)
            batch_size = {'train': args.train_batch_size, 'val': args.val_batch_size}[k]
            batches = datasets.questions.batch.batch(q, batch_size)
            batches = batches.cache()
            questions[k] = batches
            problems_with_questions[k] = [pp for pp in map(py_str, p) if pp in questions_all]
            logging.info(f'Number of {k} problems with questions: {len(problems_with_questions[k])}')

        checkpoint_dir = os.path.join(output, 'tf_ckpts')
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
        tensorboard = callbacks.TensorBoard(log_dir=log_dir, profile_batch=args.profile_batch, histogram_freq=1,
                                            embeddings_freq=1)
        cbs = [
            tensorboard,
            callbacks.Time(problems={k: next(iter(v.take(32).batch(32))) for k, v in problems.items() if
                                     cardinality_finite(v) > 0},
                           tensorboard=tensorboard),
            tf.keras.callbacks.CSVLogger(os.path.join(output, 'epochs.csv')),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(epoch_ckpt_dir, 'weights.{epoch:05d}-{val_binary_accuracy:.2f}.tf'),
                save_weights_only=True, verbose=0),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(acc_ckpt_dir, 'weights.{epoch:05d}-{val_binary_accuracy:.2f}.tf'),
                save_weights_only=True, verbose=1, monitor='val_binary_accuracy', save_best_only=True)
        ]

        symbol_cost_evaluation_callback = None
        if args.solver_eval_start is not None or args.solver_eval_step is not None:
            solver_eval_problems = problems['val']
            if args.solver_eval_val_problems is not None and args.solver_eval_val_problems >= 0:
                solver_eval_problems = solver_eval_problems.take(args.solver_eval_val_problems)
            if args.solver_eval_train_without_questions:
                solver_eval_problems_train = problems['train']
            else:
                solver_eval_problems_train = tf.data.Dataset.from_tensor_slices(problems_with_questions['train'])
            if args.solver_eval_train_problems is not None and args.solver_eval_train_problems >= 0:
                solver_eval_problems_train = solver_eval_problems_train.take(args.solver_eval_train_problems)
            if cardinality_finite(solver_eval_problems_train, 1) >= 1:
                solver_eval_problems = solver_eval_problems.concatenate(solver_eval_problems_train)
            solver_eval_problems = list(OrderedSet(map(py_str, solver_eval_problems)))
            problems_to_graphify.update(solver_eval_problems)

            problem_categories = {'all': None, 'with_questions': questions_all.keys()}
            for cat_name, cat_filename in args.problem_set:
                with open(cat_filename) as f:
                    problem_categories[cat_name] = [l.rstrip('\n') for l in f]

            symbol_cost_evaluation_callback = callbacks.SymbolCostEvaluation(
                os.path.join(output, 'epochs_solver_eval.csv'),
                solver=solver,
                problems=solver_eval_problems,
                symbol_type=args.symbol_type,
                splits={k: list(map(py_str, v)) for k, v in problems.items()},
                batch_size=args.solver_eval_batch_size,
                start=args.solver_eval_start,
                step=args.solver_eval_step,
                iterations=args.solver_eval_iterations,
                output_dir=output,
                tensorboard=tensorboard,
                problem_categories=problem_categories,
                baseline=args.symbol_cost_model == 'baseline',
                parallel=parallel,
                train_without_questions=args.solver_eval_train_without_questions)
            cbs.append(symbol_cost_evaluation_callback)

        logging.info(f'Symbol cost model: {args.symbol_cost_model}')
        if args.symbol_cost_model == 'baseline':
            model_symbol_cost = models.symbol_cost.Baseline()
        else:
            if args.symbol_cost_model == 'direct':
                model_symbol_cost = models.symbol_cost.Direct(questions_all)
            elif args.symbol_cost_model == 'composite':
                embedding_to_cost = None
                logging.info(f'Symbol embedding model: {args.symbol_embedding_model}')
                if args.symbol_embedding_model == 'simple':
                    model_symbol_embedding = models.symbol_features.Simple(clausifier, args.symbol_type)
                    if args.embedding_to_cost_hidden_layer is None:
                        cbs.append(callbacks.Weights(tensorboard))
                    if args.simple_model_kernel is not None:
                        kernel = np.fromstring(args.simple_model_kernel, count=model_symbol_embedding.n, sep=',')
                        logging.info(f'Simple model kernel: {kernel}')
                        embedding_to_cost = tf.keras.layers.Dense(1, use_bias=False, trainable=False,
                                                                  kernel_initializer=tf.constant_initializer(kernel))

                elif args.symbol_embedding_model == 'gcn':
                    graphifier = Graphifier(clausifier, max_number_of_nodes=args.max_num_nodes)
                    # problems_to_graphify = set(map(py_str, problems_all))
                    graphs, graphs_df = get_graphs(graphifier, problems_to_graphify)
                    for problem_name, rec in graphs_df.iterrows():
                        problem_records[problem_name].update(rec.to_dict())
                    logging.info(f'Number of problems graphified: {len(graphs)}')
                    save_df(graphs_df, os.path.join(output, 'graphs'))

                    model_symbol_embedding = models.symbol_features.Graph(graphifier, graphs, args.symbol_type,
                                                                          embedding_size=args.gcn_message_size,
                                                                          num_layers=args.gcn_depth,
                                                                          activation=args.gcn_activation,
                                                                          conv_norm=args.conv_norm,
                                                                          residual=not args.no_residual,
                                                                          layer_norm=not args.no_layer_norm,
                                                                          dropout=args.gcn_dropout,
                                                                          symbol_types=[args.symbol_type])
                else:
                    raise ValueError(f'Unsupported symbol embedding model: {args.symbol_embedding_model}')
                if embedding_to_cost is None:
                    if args.embedding_to_cost_hidden_layer is None:
                        embedding_to_cost = tf.keras.layers.Dense(1, name='embedding_to_cost',
                                                                  kernel_regularizer=tf.keras.regularizers.L1L2(
                                                                      l1=args.embedding_to_cost_l1,
                                                                      l2=args.embedding_to_cost_l2))
                    else:
                        embedding_to_cost = tf.keras.Sequential([
                            tf.keras.layers.Dense(args.embedding_to_cost_hidden_layer,
                                                  activation='relu',
                                                  kernel_regularizer=tf.keras.regularizers.L1L2(
                                                      l1=args.embedding_to_cost_l1,
                                                      l2=args.embedding_to_cost_l2)),
                            tf.keras.layers.Dense(1,
                                                  kernel_regularizer=tf.keras.regularizers.L1L2(
                                                      l1=args.embedding_to_cost_l1,
                                                      l2=args.embedding_to_cost_l2))
                        ], name='embedding_to_cost')
                model_symbol_cost = models.symbol_cost.Composite(model_symbol_embedding, embedding_to_cost,
                                                                 l2=args.symbol_cost_l2)
            else:
                raise ValueError(f'Unsupported symbol cost model: {args.symbol_cost_model}')

        save_df(dataframe_from_records(list(problem_records.values()), index_keys='name', dtypes=problem_records_types),
                os.path.join(output, 'problems'))

        model_logit = models.question_logit.QuestionLogitModel(model_symbol_cost)

        Optimizer = {
            'sgd': tf.keras.optimizers.SGD,
            'adam': tf.keras.optimizers.Adam,
            'rmsprop': tf.keras.optimizers.RMSprop
        }[args.optimizer]
        optimizer = Optimizer(learning_rate=args.learning_rate)

        model_logit.compile(optimizer=optimizer)

        if args.restore_checkpoint is not None:
            model_logit.load_weights(args.restore_checkpoint)

        # We need to set_model before we begin using tensorboard. Tensorboard is used in other callbacks in symbol cost evaluation.
        tensorboard.set_model(model_logit)

        print('Initial evaluation...')
        if symbol_cost_evaluation_callback is not None and symbol_cost_evaluation_callback.start <= -1:
            print('Evaluating symbol cost model before first training epoch...')
            logs = symbol_cost_evaluation_callback.evaluate(symbol_cost_model=model_symbol_cost, epoch=-1)
            print(logs)

        if not isinstance(model_symbol_cost, models.symbol_cost.Baseline):
            for k, x in questions.items():
                print(f'Evaluating logit model on {k} questions...')
                model_logit.evaluate(x)

            if args.initial_evaluation_extra:
                initial_evaluation(model_logit, questions_all, problems_all, args.train_batch_size)

            if args.epochs >= 1:
                print('Training...')
                model_logit.fit(questions['train'], validation_data=questions['val'], epochs=args.epochs, callbacks=cbs)


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
