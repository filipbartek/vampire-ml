#!/usr/bin/env python3.7

import argparse
import logging
import sys

import numpy as np

import action_compare
import action_fit
import action_vampire

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # SWV567-1.014.p has clause depth of more than the default recursion limit of 1000,
    # making `json.load()` raise `RecursionError`.
    sys.setrecursionlimit(2000)
    np.seterr(all='raise')

    parser = argparse.ArgumentParser()
    # Why does dest need to be specified? See https://bugs.python.org/issue29298
    subparsers = parser.add_subparsers(help='action', dest='action', required=True)

    action_vampire.add_arguments(subparsers.add_parser('vampire', aliases=['v']))
    action_compare.add_arguments(subparsers.add_parser('compare', aliases=['c']))
    action_fit.add_arguments(subparsers.add_parser('fit', aliases=['f']))

    # TODO: Allow loading a trained model.

    namespace = parser.parse_args()
    namespace.action(namespace)
