import json
import logging
import os

import numpy as np
import pyparsing
import scipy

from questions.memory import memory
from utils import is_compatible
from weight import vampire

log = logging.getLogger(__name__)


@memory.cache(ignore=['signature'])
def load(path, signature, max_size=None):
    log.debug(f'Loading proof: {path}')
    with open(os.path.join(path, 'meta.json')) as f:
        meta = json.load(f)
    samples = None
    if meta['szs_status'] in ['THM', 'CAX', 'UNS']:
        # We only care about successful refutation proofs.
        stdout_path = os.path.join(path, 'stdout.txt')
        symbol_name_to_index = {s: i for i, s in enumerate(signature)}
        try:
            # Raises `RuntimeError` if the output file is too large.
            # Raises `ValueError` if no proof is found in the output file.
            samples = load_proof_samples(stdout_path, symbol_name_to_index, max_size)
        except (RuntimeError, ValueError) as e:
            log.warning(f'{stdout_path}: Failed to load proof: {str(e)}')
    return meta['problem'], samples


def load_proof_samples(stdout_path, signature, max_size=None):
    if max_size is not None:
        cur_size = os.path.getsize(stdout_path)
        if cur_size > max_size:
            raise RuntimeError(f'Proof file is too large: {cur_size} > {max_size}')
    with open(stdout_path) as f:
        stdout = f.read()
    # Raises `ValueError` if no proof is found.
    df_formulas, df_operations = vampire.formulas.extract_df(stdout, roles=['proof', 'active'])
    formula_sets = {role: df_operations[df_operations.role == role].formula_id for role in ['active', 'proof']}
    assert all(ids.nunique() == len(ids) for ids in formula_sets.values())
    assert all(set(ids) <= set(df_formulas.index) for ids in formula_sets.values())
    df_formulas_active = df_formulas.loc[formula_sets['active']]
    df_formulas_active['role_proof'] = df_formulas_active.index.isin(formula_sets['proof'])
    samples_list = list(df_to_samples(df_formulas_active, signature))
    if len(samples_list) == 0:
        raise RuntimeError(f'{stdout_path}: No proof samples were extracted.')
    samples_aggregated = {
        'token_counts': scipy.sparse.vstack(s['token_counts'] for s in samples_list),
        'proof': scipy.sparse.csc_matrix([[s['proof']] for s in samples_list], dtype=bool),
        'goal': scipy.sparse.csc_matrix([[s['goal']] for s in samples_list], dtype=bool)
    }
    return samples_aggregated


def df_to_samples(df, signature):
    for index, row in df.iterrows():
        try:
            yield row_to_sample(index, row, signature)
        except RuntimeError as e:
            log.debug(str(e))
            continue


def row_to_sample(index, row, signature):
    # Raises `RuntimeError` if parsing of the formula fails.
    formula = row.string
    proof = row.role_proof
    proof_symbol = '-+'[proof]
    formula_description = f'{proof_symbol} {index}: {formula}'
    try:
        # Raises `pyparsing.ParseException` if parsing of the formula fails.
        # Raises `RecursionError` if the formula is too deep.
        token_counts = vampire.clause.token_counts(formula)
    except (pyparsing.ParseException, RecursionError) as e:
        raise RuntimeError(f'{formula_description}. Failed to parse.') from e
    return {
        'token_counts': occurrence_count_vector(token_counts, signature),
        'proof': proof,
        'goal': row.extra_goal
    }


def occurrence_count_vector(token_counts, symbol_name_to_index, dtype=np.uint32):
    """
    0. variable
    1. negation
    2. equality
    3. inequality
    4:. symbols
    """
    data = [sum(token_counts['variable']), *(token_counts[k] for k in ['negation', 'equality', 'inequality'])]
    if any(d > 0 for d in data):
        data, indices = map(list, zip(*((d, i) for i, d in enumerate(data) if d != 0)))
    else:
        data, indices = [], []
    data += token_counts['symbol'].values()
    assert is_compatible(data, dtype)
    indices += [4 + symbol_name_to_index[s] for s in token_counts['symbol'].keys()]
    assert len(data) == len(indices)
    indptr = [0, len(data)]
    result = scipy.sparse.csr_matrix((data, indices, indptr), shape=(1, 4 + len(symbol_name_to_index)), dtype=dtype)
    return result
