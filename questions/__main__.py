#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import json
import logging
import os

import joblib
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from proving.graphifier import Graphifier
from proving.solver import Solver
from questions import datasets
from questions import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', nargs='*')
    parser.add_argument('--questions-dir')
    parser.add_argument('--questions-file')
    parser.add_argument('--max-problems', type=int, default=None)
    parser.add_argument('--logs-dir', default='logs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-level', choices=['INFO', 'DEBUG'], default='INFO')
    parser.add_argument('--validation-split', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--symbol-type', choices=['predicate', 'function'], default='predicate')
    parser.add_argument('--solver-evaluation-start', type=int, default=None)
    parser.add_argument('--solver-evaluation-step', type=int, default=None)
    parser.add_argument('--profile-batch', default=0)
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--run-eagerly', action='store_true')
    parser.add_argument('--symbol-embedding-model', default='gcn', choices=['simple', 'gcn'])
    parser.add_argument('--cache-dir')
    parser.add_argument('--cache-mem', action='store_true')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--max-num-nodes', type=int, default=100000)
    parser.add_argument('--preload-graphs', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(args.run_eagerly)
    tf.summary.experimental.set_step(0)

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

    solver = Solver(timeout=20)

    with joblib.parallel_backend('threading', n_jobs=args.jobs):
        with writer_train.as_default():
            questions_all = datasets.questions.load_questions.load(args.questions_file, args.questions_dir)

        problems_all = datasets.problems.get_dataset(patterns)
        logging.info('Number of problems available: %d', problems_all.cardinality())
        logging.debug('Leading 10 problems: %s', [bytes.decode(p.numpy()) for p in problems_all.take(10)])
        if args.max_problems is not None:
            problems_all = problems_all.take(args.max_problems)
        logging.info('Number of problems taken: %d', problems_all.cardinality())
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
        for k in problems:
            logging.info(f'Number of {k} problems: %d', problems[k].cardinality())

        cache_dir = args.cache_dir
        if cache_dir is not None:
            # Cache identification parameters:
            # - problem sets (patterns, validation_split, max_problems)
            # - question set (question_dir)
            # - dataset name (validation or train)
            # We only hash the parameters that cannot be easily represented by a string.
            hash_data = json.dumps({
                'patterns': patterns,
                'validation_split': args.validation_split
            }).encode()
            hash_digest = hashlib.md5(hash_data).hexdigest()
            cache_dir = os.path.join(cache_dir, str(args.max_problems), hash_digest)

        questions = {}
        problems_with_questions = set()
        for k, p in problems.items():
            q = datasets.questions.individual.dict_to_dataset(questions_all, p)
            if cache_dir is not None:
                cache_path = os.path.join(cache_dir, k)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                logging.info('Caching into: %s', cache_path)
                q = q.cache(cache_path)
            problems_with_questions.update(bytes.decode(qq['problem'].numpy()) for qq in q)
            batches = datasets.questions.batch.batch(q, args.batch_size)
            if args.cache_mem:
                batches = batches.cache()
            questions[k] = batches

        logging.info(f'Symbol embedding model: {args.symbol_embedding_model}')
        if args.symbol_embedding_model == 'simple':
            model_symbol_embedding = models.symbol_features.simple.SimpleSymbolFeaturesModel(solver, args.symbol_type)
        elif args.symbol_embedding_model == 'gcn':
            graphifier = Graphifier(solver, max_number_of_nodes=args.max_num_nodes)
            graphs = graphifier.problems_to_graphs_dict(problems_with_questions)
            model_symbol_embedding = models.symbol_features.graph.GraphSymbolFeatures(graphifier, graphs,
                                                                                      args.symbol_type, num_layers=4)
        else:
            raise ValueError(f'Unsupported symbol embedding model: {args.symbol_embedding_model}')

        if args.preload_graphs:
            for k, v in problems.items():
                # TODO: Ensure that the batches match the question batches exactly.
                with tqdm(v.batch(args.batch_size), unit='batch', desc=f'Preloading {k} graphs') as t:
                    stats = {'total': 0, 'successes': 0, 'failures': 0}
                    t.set_postfix(stats)
                    for batch in t:
                        res = model_symbol_embedding.call(batch)
                        cur_successes = tf.math.count_nonzero(res['valid']).numpy()
                        stats['total'] += len(batch)
                        stats['successes'] += cur_successes
                        stats['failures'] += len(batch) - cur_successes
                        t.set_postfix(stats)

        model_symbol_cost = models.symbol_cost.SymbolCostModel(model_symbol_embedding)
        model_symbol_cost.compile(metrics=[models.symbol_cost.SolverSuccessRate(solver, args.symbol_type)])
        model_logit = models.question_logit.QuestionLogitModel(model_symbol_cost)
        model_logit.compile(optimizer=args.optimizer,
                            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0),
                                     tf.keras.metrics.BinaryCrossentropy(from_logits=True)])

        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=args.profile_batch),
            models.question_logit.SymbolCostEvaluationCallback(problems=problems['train'].batch(1),
                                                               problems_validation=problems['validation'].batch(1),
                                                               start=args.solver_evaluation_start,
                                                               step=args.solver_evaluation_step)
        ]
        logging.info('Training...')
        fit_res = model_logit.fit(questions['train'], validation_data=questions['validation'], epochs=args.epochs,
                                  callbacks=callbacks)
        print(fit_res)


if __name__ == '__main__':
    main()
