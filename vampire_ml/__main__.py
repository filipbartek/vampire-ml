#!/usr/bin/env python3

import argparse
import logging
import logging.config
import sys

import joblib
import numpy as np

from vampire_ml import action_vampire
from vampire_ml import fit


def logging_basic_config(level=logging.INFO):
    if isinstance(level, str):
        level = getattr(logging, level)
    logging.basicConfig(level=level)
    if logging.getLogger().level != level:
        logging.warning('Failed to configure the logging level. Desired: %s. Actual: %s.',
                        level, logging.getLogger().level)


if __name__ == '__main__':
    np.seterr(all='raise', under='warn')

    parser = argparse.ArgumentParser(prog='python -m vampire_ml',
                                     description='Recognized environment variables: TPTP (path to TPTP), TPTP_PROBLEMS (path to TPTP/Problems), SCRATCH (path to scratch directory), $XDG_CACHE_HOME')
    # Why does dest need to be specified? See https://bugs.python.org/issue29298
    subparsers = parser.add_subparsers(help='action', dest='action', required=True)

    action_vampire.add_arguments(subparsers.add_parser('vampire', aliases=['v']))
    fit.add_arguments(subparsers.add_parser('fit', aliases=['f']))

    parser.add_argument('--log-level', choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Default logging level')
    parser.add_argument('--log-config', default='logging.conf')
    parser.add_argument('--log-output', default='vampire_ml.log')
    parser.add_argument('--jobs', '-j', type=int, default=1,
                        help='Maximum number of concurrently running jobs. If -1 all CPUs are used.')

    # TODO: Allow loading a trained model.

    namespace = parser.parse_args()

    logging_basic_config(level=namespace.log_level)
    try:
        logging.config.fileConfig(namespace.log_config, defaults={'filename': namespace.log_output},
                                  disable_existing_loggers=False)
    except KeyError:
        logging.warning('Failed to load log config file: %s', namespace.log_config)
    logging.info('Beginning.')
    logging.info('Arguments: %s', sys.argv)
    logging.debug('Logging config file: %s', namespace.log_config)
    logging.info('Logging output file: %s', namespace.log_output)
    logging.info('Logging default level: %s', logging.getLogger().level)

    logging.getLogger('matplotlib').setLevel(logging.INFO)

    np.random.seed(0)

    with joblib.parallel_backend('threading', n_jobs=namespace.jobs):
        namespace.action(namespace)
