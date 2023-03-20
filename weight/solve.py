import joblib
import logging
import sys
import warnings

import hydra
import pandas as pd
import tensorflow as tf
from omegaconf import OmegaConf

import classifier
from dense import Dense
from questions import models
from questions.graphifier import Graphifier
from questions.solver import Solver
from train import evaluate_options
from train import get_problems

log = logging.getLogger(__name__)


@hydra.main(config_path='.', config_name='config', version_base='1.1')
def main(cfg):
    clausifier = Solver(options=OmegaConf.to_container(cfg.options.common, resolve=True))
    model_logit = create_model(cfg, clausifier)

    ckpt = tf.train.Checkpoint(model_logit)

    model = model_logit.symbol_weight_model
    
    if cfg.checkpoint.restore is not None:
        restore_path = hydra.utils.to_absolute_path(cfg.checkpoint.restore)
        ckpt.restore(restore_path).expect_partial()
        log.info(f'Restored checkpoint: {restore_path}')
    else:
        log.info('No checkpoint specified.')

    problems, problem_names = get_problems(cfg.problem)
    if len(problems) == 0:
        warnings.warn('No problems specified.')
        return

    model_result = model.predict(problems, batch_size=cfg.batch.size)

    parallel = joblib.Parallel(verbose=1)
    df = evaluate_options(model_result, problems, clausifier, cfg, cfg.options.evaluation.default, parallel, out_dir='eval')
        
    print(df)


def create_model(cfg, clausifier, cache=False):
    graphifier = Graphifier(clausifier, max_number_of_nodes=cfg.max_problem_nodes, cache=cache)

    output_ntypes = ['predicate', 'function', 'variable', 'atom', 'equality']
    # Per-node values:
    # - symbol <- predicate, function
    # Readouts:
    # - variable_occurrence <- variable
    # - variable_count <- variable
    # - literal_positive <- atom, equality
    # - literal_negative <- atom, equality
    # - equality <- equality
    # - inequality <- equality
    # - number <- function
    gcn = models.symbol_features.GCN(cfg.gcn, graphifier.canonical_etypes, graphifier.ntype_in_degrees,
                                     graphifier.ntype_feat_sizes, output_ntypes=output_ntypes)
    # Outputs an embedding for each token.
    model_symbol_embedding = models.symbol_features.Graph(graphifier, gcn)
    embedding_to_weight = {
        name: Dense(1, name=name, activation=cfg.embedding_to_cost.activation,
                    output_bias=cfg.embedding_to_cost.output_bias,
                    kernel_regularizer=tf.keras.regularizers.L1L2(**cfg.embedding_to_cost.regularization)) for
        name in cfg.clause_features + ['symbol']
    }
    model_symbol_weight = models.symbol_cost.Composite(model_symbol_embedding, embedding_to_weight,
                                                       l2=cfg.symbol_cost.l2)
    model_logit = classifier.Classifier(model_symbol_weight)
    return model_logit


if __name__ == '__main__':
    with pd.option_context('display.max_columns', sys.maxsize,
                           'display.width', None,
                           'display.float_format', '{:.2f}'.format):
        main()
