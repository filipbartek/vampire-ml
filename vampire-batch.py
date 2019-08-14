#!/usr/bin/env python3.7

import argparse
import logging

from action_fit import ActionFit
from action_stats import ActionStats
from action_vampire import ActionVampire

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='action', required=True)

    parser_vampire = subparsers.add_parser('vampire', aliases=['v', 'collect', 'evaluate'])
    vampire = ActionVampire(parser_vampire)

    parser_stats = subparsers.add_parser('stats', aliases=['s'])
    stats = ActionStats(parser_stats)

    parser_fit = subparsers.add_parser('fit', aliases=['f'])
    fit = ActionFit(parser_fit)

    # TODO: Allow loading a trained model.

    namespace = parser.parse_args()
    namespace.action(namespace)
