import logging

import hydra
import numpy as np
import tensorflow as tf
from attributedict.collections import AttributeDict

from questions import callbacks
from questions import datasets
from questions import models
from questions import param


def run(params, state):
    logging.info(f'Symbol cost model: {params.symbol_cost.model}')
    if params.symbol_cost.model == 'baseline':
        model_symbol_cost = models.symbol_cost.Baseline()
    elif params.symbol_cost.model == 'direct':
        model_symbol_cost = models.symbol_cost.Direct(state.questions_all)
    elif params.symbol_cost.model == 'composite':
        model_symbol_cost = get_composite_symbol_cost_model(params, state)
    else:
        raise ValueError(f'Unsupported symbol cost model: {params.symbol_cost.model}')

    if state.symbol_cost_evaluation_callback is not None and state.symbol_cost_evaluation_callback.start <= -1:
        print('Evaluating symbol cost model before first training epoch...')
        logs = state.symbol_cost_evaluation_callback.evaluate(symbol_cost_model=model_symbol_cost, epoch=-1)
        print(logs)

    if not isinstance(model_symbol_cost, models.symbol_cost.Baseline):
        model_logit = models.question_logit.QuestionLogitModel(model_symbol_cost)

        Optimizer = {
            'sgd': tf.keras.optimizers.SGD,
            'adam': tf.keras.optimizers.Adam,
            'rmsprop': tf.keras.optimizers.RMSprop
        }[params.optimizer]
        optimizer = Optimizer(learning_rate=params.learning_rate)

        model_logit.compile(optimizer=optimizer)

        if params.load_checkpoint is not None:
            model_logit.load_weights(hydra.utils.to_absolute_path(params.load_checkpoint))
            logging.info(f'Checkpoint loaded: {params.load_checkpoint}')

        # We need to set_model before we begin using tensorboard. Tensorboard is used in other callbacks in symbol cost evaluation.
        state.tensorboard.set_model(model_logit)

        if params.initial_eval:
            print('Initial evaluation of question logit model...')
            for k in state.question_batches:
                print(f'Evaluating logit model on {k} questions...')
                if k == 'train':
                    x = datasets.questions.batch.batch(state.questions[k], params.batch_size.val)
                else:
                    x = state.question_batches[k]
                metrics = model_logit.evaluate(x, return_dict=True)
                print(metrics)

        if params.epochs >= 1:
            print('Training...')
            model_logit.fit(state.question_batches['train'], validation_data=state.question_batches['val'],
                            epochs=params.epochs, callbacks=state.cbs)


def get_composite_symbol_cost_model(args, state):
    model_symbol_embedding = get_symbol_embedding_model(args, state)
    if isinstance(model_symbol_embedding, models.symbol_features.Simple) and args.simple_model_kernel is not None:
        kernel = np.fromstring(args.simple_model_kernel, count=model_symbol_embedding.n, sep=',')
        logging.info(f'Simple model kernel: {kernel}')
        embedding_to_cost = tf.keras.layers.Dense(1, use_bias=False, trainable=False,
                                                  kernel_initializer=tf.constant_initializer(kernel))
    else:
        if args.embedding_to_cost.hidden.units == 0:
            embedding_to_cost = tf.keras.layers.Dense(1, name='embedding_to_cost',
                                                      kernel_regularizer=tf.keras.regularizers.L1L2(
                                                          l1=args.embedding_to_cost.l1,
                                                          l2=args.embedding_to_cost.l2))
        else:
            embedding_to_cost = tf.keras.Sequential([
                tf.keras.layers.Dense(args.embedding_to_cost.hidden.units,
                                      activation=args.embedding_to_cost.hidden.activation,
                                      kernel_regularizer=tf.keras.regularizers.L1L2(
                                          l1=args.embedding_to_cost.l1,
                                          l2=args.embedding_to_cost.l2)),
                tf.keras.layers.Dense(1,
                                      kernel_regularizer=tf.keras.regularizers.L1L2(
                                          l1=args.embedding_to_cost.l1,
                                          l2=args.embedding_to_cost.l2))
            ], name='embedding_to_cost')
    return models.symbol_cost.Composite(model_symbol_embedding, embedding_to_cost, l2=args.symbol_cost.l2)


def get_symbol_embedding_model(args, state):
    logging.info(f'Symbol embedding model: {args.symbol_embedding_model}')
    if args.symbol_embedding_model == 'simple':
        model_symbol_embedding = models.symbol_features.Simple(state.clausifier, args.symbol_type)
        if args.embedding_to_cost_hidden_layer is None:
            state.cbs.append(callbacks.Weights(state.tensorboard))
    elif args.symbol_embedding_model == 'gcn':
        constraint = None
        if args.gcn.max_norm is not None:
            constraint = tf.keras.constraints.max_norm(args.gcn.max_norm)
        gcn = models.symbol_features.GCN(state.graphifier.canonical_etypes,
                                         state.graphifier.ntype_in_degrees,
                                         state.graphifier.ntype_feat_sizes,
                                         args.gcn,
                                         output_ntypes=[args.symbol_type],
                                         constraint=constraint)
        model_symbol_embedding = models.symbol_features.Graph(state.graphifier.empty_graph(), state.graphs,
                                                              args.symbol_type, gcn)
    else:
        raise ValueError(f'Unsupported symbol embedding model: {args.symbol_embedding_model}')
    return model_symbol_embedding


def param_regularizer(name, default=0):
    return param.Float(name, 0, 1, default=default)


def param_activation(name):
    return param.Categorical(name, ['relu', 'sigmoid'], default='relu')


# https://neptune.ai/blog/hyperparameter-tuning-on-any-python-script
space = AttributeDict({
    'batch_size': {
        'train': param.Int('batch_size_train', 1, 256, log=True, default=128),
        'val': param.Int('batch_size_val', 1, 256, log=True, default=128)
    },
    'optimizer': param.Categorical('optimizer', ['sgd', 'adam', 'rmsprop'], default='adam'),
    'learning_rate': param.Float('learning_rate', 0.0001, 1.0, log=True, default=0.001),
    'symbol_embedding_model': param.Categorical('symbol_embedding_model', ['gcn', 'simple'], default='gcn'),
    'symbol_cost': {
        'model': param.Categorical('symbol_cost_model', ['composite', 'baseline', 'direct'], default='composite',
                                   searchable=False),
        'l2': param_regularizer('symbol_cost_l2'),
    },
    'embedding_to_cost': {
        'hidden': {
            'units': param.Int('embedding_to_cost_hidden_units', 0, 1024, default=0),
            'activation': param_activation('embedding_to_cost_hidden_activation')
        },
        'l1': param_regularizer('embedding_to_cost_l1'),
        'l2': param_regularizer('embedding_to_cost_l2')
    },
    'gcn': {
        'depth': param.Int('gcn_depth', 1, 64, log=True, default=4),
        'message_size': param.Int('gcn_message_size', 1, 256, log=True, default=16),
        'activation': param_activation('gcn_activation'),
        'aggregate': param.Categorical('gcn_aggregate', ['sum', 'max', 'min', 'mean'], default='sum'),
        'dropout': {
            'input': param_regularizer('gcn_dropout_input'),
            'hidden': param_regularizer('gcn_dropout_hidden')
        },
        'max_norm': param.Float('gcn_max_norm', 0, 5, default=0),
        'layer_norm': param.Boolean('gcn_layer_norm', default=True),
        'residual': param.Boolean('gcn_residual', default=True),
        'conv_norm': param.Categorical('gcn_conv_norm', ['both', 'right', 'none'], default='both')
    }
})
