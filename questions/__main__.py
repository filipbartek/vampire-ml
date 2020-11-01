#!/usr/bin/env python3

import logging
import os

import tensorflow as tf

from proving.solver import Solver
from questions import datasets
from questions import models


def main():
    logging.basicConfig(level=logging.INFO)
    tf.random.set_seed(0)
    tf.config.run_functions_eagerly(True)

    patterns = ['**/*-*.p', '**/*+*.p']
    validation_split = 0.5
    max_problems = 100
    question_dir = '/home/filip/projects/vampire-ml/questions/pred_10k'
    batch_size = 10
    symbol_type = 'predicate'

    solver = Solver(timeout=20)

    problems = datasets.problems.get_datasets_split(patterns, validation_split, max_problems)
    questions = datasets.questions.get_datasets_batched(question_dir, problems, batch_size,
                                                        os.path.join('cache', 'questions'))
    for k, q in questions.items():
        logging.info(f'Preloading {k} batches...')
        datasets.questions.preload(q)

    model_simple = models.symbol_features.SimpleSymbolFeaturesModel(solver, symbol_type)
    model_symbol_cost = models.symbol_cost.SymbolCostModel(model_simple)
    model_symbol_cost.compile(metrics=[models.symbol_cost.SolverSuccessRate(solver, symbol_type)], run_eagerly=True)
    model_logit = models.question_logit.QuestionLogitModel(model_symbol_cost)
    model_logit.compile(optimizer='adam',
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0),
                                 tf.keras.metrics.BinaryCrossentropy(from_logits=True)],
                        run_eagerly=True)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch='10,20'),
        models.question_logit.SymbolCostEvaluationCallback(problems=problems['train'].batch(1),
                                                           problems_validation=problems['validation'].batch(1),
                                                           step=10)
    ]
    logging.info('Training...')
    fit_res = model_logit.fit(questions['train'], validation_data=questions['validation'], epochs=100,
                              callbacks=callbacks)
    print(fit_res)


if __name__ == '__main__':
    main()
