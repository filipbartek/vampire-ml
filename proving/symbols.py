import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def symbols_of_type(symbols, symbol_type):
    assert symbol_type in {'predicate', 'function'}
    if symbol_type == 'predicate':
        symbols = symbols[symbols.index.get_level_values('isFunction') == False]
    if symbol_type == 'function':
        symbols = symbols[symbols.index.get_level_values('isFunction') == True]
    symbols.index = symbols.index.droplevel(0)
    return symbols


def load(file):
    # Throws FileNotFoundError if `file` does not exist.
    log.debug(f'Loading {file} of size {os.path.getsize(file)}.')
    # The column 'name' may contain single quoted strings.
    # See http://www.tptp.org/TPTP/SyntaxBNF.html
    # <fof_plain_term> ::= <functor> ::= <atomic_word> ::= <single_quoted> ::= <single_quote> ::: [']
    # We assume that there are no NAs in the symbols CSV table.
    # Note that for example in SWV478+2.p there is a symbol called 'null' that may alias with the NA filtering
    # (its name being misinterpreted as a missing value).
    return pd.read_csv(file, index_col=['isFunction', 'id'], quotechar='\'', escapechar='\\', na_filter=False,
                       dtype={
                           'isFunction': np.bool,
                           'id': pd.UInt32Dtype(),
                           'name': 'object',
                           'arity': pd.UInt32Dtype(),
                           'usageCnt': pd.UInt32Dtype(),
                           'unitUsageCnt': pd.UInt32Dtype(),
                           'inGoal': np.bool,
                           'inUnit': np.bool,
                           'skolem': np.bool,
                           'inductionSkolem': np.bool
                       })


def save(symbols, file):
    symbols.to_csv(file, quotechar='\'', escapechar='\\')
