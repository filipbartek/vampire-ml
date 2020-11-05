#!/usr/bin/env python3

import argparse
import logging

import joblib
import tensorflow as tf
from tqdm import tqdm

from proving.graphifier import Graphifier
from proving.solver import Solver
from questions import datasets
from questions import models


def get_symbol_embedding_model(model_type, solver, symbol_type, max_number_of_nodes=None):
    logging.info(f'Symbol embedding model: {model_type}')
    if model_type == 'simple':
        return models.symbol_features.simple.SimpleSymbolFeaturesModel(solver, symbol_type)
    elif model_type == 'gcn':
        graphifier = Graphifier(solver, max_number_of_nodes=max_number_of_nodes)
        return models.symbol_features.graph.GraphSymbolFeatures(graphifier, symbol_type)
    else:
        raise ValueError(f'Unsupported symbol embedding model: {model_type}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', nargs='*')
    parser.add_argument('--questions-dir')
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
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(args.run_eagerly)

    patterns = args.problem
    if patterns is None or len(patterns) == 0:
        patterns = ['**/*-*.p', '**/*+*.p']
        logging.info('Defaulting problem patterns to: %s', patterns)

    solver = Solver(timeout=20)

    with joblib.parallel_backend('threading', n_jobs=args.jobs):
        problems, questions = datasets.common.get_datasets(patterns, args.validation_split, args.max_problems,
                                                           args.questions_dir, args.batch_size, args.cache_dir,
                                                           args.cache_mem)

        model_symbol_embedding = get_symbol_embedding_model(args.symbol_embedding_model, solver, args.symbol_type,
                                                            args.max_num_nodes)

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
            tf.keras.callbacks.TensorBoard(log_dir=args.logs_dir, profile_batch=args.profile_batch),
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
