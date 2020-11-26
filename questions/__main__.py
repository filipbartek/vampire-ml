#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import json
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

from proving.graphifier import Graphifier
from proving.memory import memory
from proving.solver import Solver
from questions import datasets
from questions import models
from questions import plot
from vampire_ml.results import save_df


def hash_digest(o):
    hash_data = json.dumps(o).encode()
    return hashlib.md5(hash_data).hexdigest()


def save_problems(problems, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.writelines(f'{bytes.decode(p.numpy())}\n' for p in problems)
    logging.info(f'List of {problems.cardinality()} problems saved: {filename}')


@memory.cache(verbose=1)
def get_graphs(graphifier, problems):
    return graphifier.problems_to_graphs_dict(problems)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', nargs='*')
    parser.add_argument('--questions-dir')
    parser.add_argument('--max-questions-per-problem', type=int)
    parser.add_argument('--max-problems', type=int, default=None)
    parser.add_argument('--logs-dir', default='logs')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--log-level', choices=['INFO', 'DEBUG'], default='INFO')
    parser.add_argument('--validation-split', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--symbol-type', choices=['predicate', 'function'], default='predicate')
    parser.add_argument('--solver-evaluation-start', type=int, default=None)
    parser.add_argument('--solver-evaluation-step', type=int, default=None)
    parser.add_argument('--profile-batch', default=0)
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--run-eagerly', action='store_true')
    parser.add_argument('--symbol-embedding-model', default='gcn', choices=['simple', 'gcn'])
    parser.add_argument('--symbol-cost-model', default='composite', choices=['composite', 'direct'])
    parser.add_argument('--simple-model-kernel')
    parser.add_argument('--cache-dir', default='cache')
    parser.add_argument('--cache-mem', action='store_true')
    parser.add_argument('--output', default='out')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--max-num-nodes', type=int, default=100000)
    parser.add_argument('--initial-evaluation-extra', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(args.run_eagerly)
    tf.summary.experimental.set_step(0)

    logging.info(f'Joblib cache location: {memory.cachedir}')

    log_dir = os.path.join(args.logs_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logging.info(f'Log directory: {log_dir}')
    writer_train = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    with writer_train.as_default():
        # https://stackoverflow.com/a/61106106/4054250
        args_series = pd.Series(args.__dict__, name='value')
        args_series.index.name = 'argument'
        tf.summary.text('args', args_series.to_markdown())

    patterns = args.problem
    if patterns is None or len(patterns) == 0:
        patterns = ['**/*-*.p', '**/*+*.p']
        logging.info('Defaulting problem patterns to: %s', patterns)
    for i, pattern in enumerate(patterns):
        if pattern[-2:] != '.p':
            patterns[i] = f'{pattern[:3]}/{pattern}.p'

    solver = Solver(timeout=20)

    with joblib.parallel_backend('threading', n_jobs=args.jobs):
        logging.info('Collecting available problems...')
        problems_all = datasets.problems.get_dataset(patterns)
        logging.info('Number of problems available: %d', problems_all.cardinality())
        save_problems(problems_all, os.path.join(args.output, 'problems', 'all.txt'))
        if args.max_problems is not None:
            problems_all = problems_all.take(args.max_problems)
        logging.info('Number of problems taken: %d', problems_all.cardinality())
        save_problems(problems_all, os.path.join(args.output, 'problems', 'taken.txt'))
        # We need to split problems first and then collect questions for each of the datasets
        # because not all problems have questions and we only generate questions samples
        # for problems with at least one question.
        assert 0 <= args.validation_split <= 1
        problems_validation_count = tf.cast(tf.cast(problems_all.cardinality(), tf.float32) * args.validation_split,
                                            tf.int64)
        problems = {
            'validation': problems_all.take(problems_validation_count),
            'train': problems_all.skip(problems_validation_count)
        }
        for k, p in problems.items():
            logging.info(f'Number of {k} problems: {p.cardinality()}')
            save_problems(p, os.path.join(args.output, 'problems', 'dataset', f'{k}.txt'))

        # TODO?: Only load questions if the batches are not cached.
        questions_file = os.path.join(args.cache_dir, f'max_questions_per_problem_{args.max_questions_per_problem}',
                                      'questions.pkl')
        with writer_train.as_default():
            # Here we load the raw, un-normalized questions (oriented element-wise differences of inverse precedences).
            questions_all = datasets.questions.load_questions.load(questions_file, args.questions_dir,
                                                                   args.max_questions_per_problem)

            question_counts = [q.shape[0] for q in questions_all.values()]
            signature_lengths = [q.shape[1] for q in questions_all.values()]

            df_index = pd.Index(questions_all.keys(), name='name')
            df = pd.DataFrame({
                'n_questions': pd.Series(question_counts, index=df_index, dtype=pd.UInt32Dtype(), name='n_questions'),
                'n_symbols': pd.Series(signature_lengths, index=df_index, dtype=pd.UInt32Dtype(), name='n_symbols')
            }, index=df_index)
            save_df(df, 'with_questions', os.path.join(args.output, 'problems'))

            tf.summary.histogram('Question counts', question_counts)
            tf.summary.histogram('Signature lengths of problems with some questions', signature_lengths)
            tf.summary.histogram('Question array sizes', [q.size for q in questions_all.values()])
            figure = plt.figure(figsize=(8, 8))
            plt.title('Problems with questions')
            sns.scatterplot(x=signature_lengths, y=question_counts)
            plt.xlabel('Symbols')
            plt.ylabel('Questions')
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig(os.path.join(args.output, 'problems', 'with_questions.png'))
            image = plot.plot_to_image(figure)
            tf.summary.image('Problems with questions', image)

        cache_patterns = os.path.join(args.cache_dir, f'patterns_{hash_digest(patterns)}')
        os.makedirs(cache_patterns, exist_ok=True)
        with open(os.path.join(cache_patterns, 'patterns.json'), 'w') as fp:
            json.dump(patterns, fp, indent=4)
        cache_dir = os.path.join(cache_patterns,
                                 f'max_problems_{args.max_problems}',
                                 f'validation_split_{args.validation_split}')

        questions = {}
        problems_with_questions = set(bytes.decode(qq['problem'].numpy()) for qq in
                                      datasets.questions.individual.dict_to_dataset(questions_all, problems_all))
        for k, p in problems.items():
            # Cache identification parameters:
            # - problem sets (patterns, validation_split, max_problems)
            # - question set (question_dir)
            # - dataset name (validation or train)
            # We only hash the parameters that cannot be easily represented by a string.
            cache_dir_dataset = os.path.join(cache_dir, k)
            logging.info(f'Caching {k} questions into: {cache_dir_dataset}')
            q = datasets.questions.individual.dict_to_dataset(questions_all, p)
            os.makedirs(cache_dir_dataset, exist_ok=True)
            q = q.cache(os.path.join(cache_dir_dataset, 'questions_individual'))
            batches = datasets.questions.batch.batch(q, args.batch_size)
            cache_dir_batches = os.path.join(cache_dir_dataset, f'batch_size_{args.batch_size}')
            os.makedirs(cache_dir_batches, exist_ok=True)
            batches = batches.cache(os.path.join(cache_dir_batches, 'question_batches'))
            if args.cache_mem:
                batches = batches.cache()
            questions[k] = batches

        logging.info(f'Symbol cost model: {args.symbol_cost_model}')
        if args.symbol_cost_model == 'direct':
            model_symbol_cost = models.symbol_cost.Direct(questions_all)
        elif args.symbol_cost_model == 'composite':
            model_symbol_cost = None
            logging.info(f'Symbol embedding model: {args.symbol_embedding_model}')
            if args.symbol_embedding_model == 'simple':
                model_symbol_embedding = models.symbol_features.simple.SimpleSymbolFeaturesModel(solver,
                                                                                                 args.symbol_type)
                if args.simple_model_kernel is not None:
                    kernel = np.fromstring(args.simple_model_kernel, count=model_symbol_embedding.n, sep=',')
                    logging.info(f'Simple model kernel: {kernel}')
                    embedding_to_cost = tf.keras.layers.Dense(1, use_bias=False, trainable=False,
                                                              kernel_initializer=tf.constant_initializer(kernel))
                    model_symbol_cost = models.symbol_cost.Composite(model_symbol_embedding, embedding_to_cost)
            elif args.symbol_embedding_model == 'gcn':
                graphifier = Graphifier(solver, max_number_of_nodes=args.max_num_nodes)
                graphs, graphs_df = get_graphs(graphifier, problems_with_questions)
                save_df(graphs_df, 'graphs', args.output)
                model_symbol_embedding = models.symbol_features.graph.GraphSymbolFeatures(graphifier, graphs,
                                                                                          args.symbol_type,
                                                                                          num_layers=4)
            else:
                raise ValueError(f'Unsupported symbol embedding model: {args.symbol_embedding_model}')
            if model_symbol_cost is None:
                model_symbol_cost = models.symbol_cost.Composite(model_symbol_embedding)
        else:
            raise ValueError(f'Unsupported symbol cost model: {args.symbol_cost_model}')
        model_symbol_cost.compile(metrics=[models.symbol_cost.SolverSuccessRate(solver, args.symbol_type)])

        model_logit = models.question_logit.QuestionLogitModel(model_symbol_cost)
        model_logit.compile(optimizer=args.optimizer)

        print('Initial evaluation...')
        for k, x in questions.items():
            eval_res = model_logit.evaluate(x, return_dict=True)
            print(f'{k}: {eval_res}')

        if args.initial_evaluation_extra:
            initial_evaluation(model_logit, questions_all, problems_all, args.batch_size)

        if args.epochs >= 1:
            callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=args.profile_batch, histogram_freq=1,
                                               embeddings_freq=1),
                models.question_logit.SymbolCostEvaluationCallback(problems=problems['train'].batch(1),
                                                                   problems_validation=problems['validation'].batch(1),
                                                                   start=args.solver_evaluation_start,
                                                                   step=args.solver_evaluation_step)
            ]
            print('Training...')
            model_logit.fit(questions['train'], validation_data=questions['validation'], epochs=args.epochs,
                            callbacks=callbacks)


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
        problem_names = [bytes.decode(p.numpy()) for p in batch['problems']]
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
