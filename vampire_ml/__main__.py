#!/usr/bin/env python3.7

import argparse
import logging

import numpy as np

from . import action_vampire

if __name__ == '__main__':
    np.seterr(all='raise', under='warn')

    parser = argparse.ArgumentParser(prog='python -m vampire_ml')
    # Why does dest need to be specified? See https://bugs.python.org/issue29298
    subparsers = parser.add_subparsers(help='action', dest='action', required=True)

    action_vampire.add_arguments(subparsers.add_parser('vampire', aliases=['v']))

    parser.add_argument('--log', choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO',
                        help='Logging level')

    # TODO: Allow loading a trained model.

    namespace = parser.parse_args()

    if namespace.log is not None:
        logging.basicConfig(level=namespace.log)
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    namespace.action(namespace)
