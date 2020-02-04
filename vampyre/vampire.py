#!/usr/bin/env python3.7

import json

import numpy as np
import pandas as pd


def load_symbols(file):
    # The column 'name' may contain single quoted strings.
    # See http://www.tptp.org/TPTP/SyntaxBNF.html
    # <fof_plain_term> ::= <functor> ::= <atomic_word> ::= <single_quoted> ::= <single_quote> ::: [']
    # We assume that there are no NAs in the symbols CSV table.
    # Note that for example in SWV478+2.p there is a symbol called 'null' that may alias with the NA filtering
    # (its name being misinterpreted as a missing value).
    return pd.read_csv(file, index_col=['isFunction', 'id'], quotechar='\'', escapechar='\\', na_filter=False,
                       dtype={
                           'isFunction': np.bool,
                           'id': pd.UInt64Dtype(),
                           'name': 'object',
                           'arity': pd.UInt64Dtype(),
                           'usageCnt': pd.UInt64Dtype(),
                           'unitUsageCnt': pd.UInt64Dtype(),
                           'inGoal': np.bool,
                           'inUnit': np.bool,
                           'skolem': np.bool,
                           'inductionSkolem': np.bool
                       })


def load_clauses(file):
    with open(file) as clauses_json_file:
        return json.load(clauses_json_file)
