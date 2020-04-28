#!/usr/bin/env python3

import argparse
import logging

import joblib
import numpy as np

from vampire_ml import action_vampire
from vampire_ml import fit

if __name__ == '__main__':
    np.seterr(all='raise', under='warn')

    parser = argparse.ArgumentParser(prog='python -m vampire_ml')
    # Why does dest need to be specified? See https://bugs.python.org/issue29298
    subparsers = parser.add_subparsers(help='action', dest='action', required=True)

    action_vampire.add_arguments(subparsers.add_parser('vampire', aliases=['v']))
    fit.add_arguments(subparsers.add_parser('fit', aliases=['f']))

    parser.add_argument('--log', choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO',
                        help='Logging level')
    parser.add_argument('--jobs', '-j', type=int, default=1,
                        help='Maximum number of concurrently running jobs. If -1 all CPUs are used.')

    # TODO: Allow loading a trained model.

    namespace = parser.parse_args()

    assert namespace.log is not None
    logging.basicConfig(format='%(relativeCreated)s - %(levelname)s - %(message)s', level=namespace.log)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    logging.info('Beginning.')

    np.random.seed(0)

    with joblib.parallel_backend('threading', n_jobs=namespace.jobs):
        namespace.action(namespace)
