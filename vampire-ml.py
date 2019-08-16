#!/usr/bin/env python3.7

import argparse
import logging

import action_fit
import action_stats
import action_vampire

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='action', required=True)

    action_vampire.add_arguments(subparsers.add_parser('vampire', aliases=['v', 'collect', 'evaluate']))
    action_stats.add_arguments(subparsers.add_parser('stats', aliases=['s']))
    action_fit.add_arguments(subparsers.add_parser('fit', aliases=['f']))

    # TODO: Allow loading a trained model.

    namespace = parser.parse_args()
    namespace.action(namespace)
