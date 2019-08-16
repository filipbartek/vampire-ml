#!/usr/bin/env python3.7

import argparse
import sys

import run_database
from lazy_csv_writer import LazyCsvWriter


def call(namespace):
    br = run_database.BatchResult(namespace.result)
    csv_writer = LazyCsvWriter(namespace.output)
    for run in br.run_list:
        if br.mode == 'vampire':
            csv_writer.writerow(run.csv_row_vampire())
        if br.mode == 'clausify':
            csv_writer.writerow(run.csv_row_clausify())


def add_arguments(parser):
    parser.set_defaults(action=call)
    parser.add_argument('result', type=str, help='result of a prove or probe run')
    parser.add_argument('--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
                        help='output CSV runs document')
