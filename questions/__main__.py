#!/usr/bin/env python3

import argparse
import logging

import tensorflow as tf

from proving.graphifier import Graphifier
from proving.solver import Solver
from questions import datasets
from questions import models


def get_symbol_embedding_model(model_type, solver, symbol_type):
    if model_type == 'simple':
        return models.symbol_features.simple.SimpleSymbolFeaturesModel(solver, symbol_type)
    elif model_type == 'gcn':
        graphifier = Graphifier(solver)
        return models.symbol_features.graph.GraphSymbolFeatures(graphifier, symbol_type)
    else:
        raise ValueError(f'Unsupported symbol embedding model: {model_type}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions-dir')
    parser.add_argument('--max-problems', type=int, default=None)
    parser.add_argument('--logs-dir', default='logs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-level', choices=['INFO', 'DEBUG'], default='INFO')
    parser.add_argument('--problems', action='append')
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
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(args.run_eagerly)

    patterns = args.problems
    if patterns is None:
        patterns = ['**/*-*.p', '**/*+*.p']
        logging.info('Defaulting problem patterns to: %s', patterns)

    problems, questions = datasets.common.get_datasets(patterns, args.validation_split, args.max_problems,
                                                       args.questions_dir, args.batch_size, args.cache_dir,
                                                       args.cache_mem)

    solver = Solver(timeout=20)

    model_symbol_embedding = get_symbol_embedding_model(args.symbol_embedding_model, solver, args.symbol_type)
    model_symbol_cost = models.symbol_cost.SymbolCostModel(model_symbol_embedding)
    model_symbol_cost.compile(metrics=[models.symbol_cost.SolverSuccessRate(solver, args.symbol_type)],
                              run_eagerly=False)
    model_logit = models.question_logit.QuestionLogitModel(model_symbol_cost)
    model_logit.compile(optimizer=args.optimizer,
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0),
                                 tf.keras.metrics.BinaryCrossentropy(from_logits=True)],
                        run_eagerly=False)

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
