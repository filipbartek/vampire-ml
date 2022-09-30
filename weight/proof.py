import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pyparsing
import scipy
import yaml

from questions.memory import memory
from utils import is_compatible
from weight import vampire

log = logging.getLogger(__name__)


@memory.cache(ignore=['parallel'], verbose=2)
def load_proofs(paths, clausifier, max_size=None, parallel=None):
    def get_signature(problem):
        try:
            # Raises `RuntimeError` when clausification fails
            return clausifier.signature(problem)
        except RuntimeError:
            return None

    def load_one(path):
        log.debug(f'Loading proof: {path}')
        with open(os.path.join(path, 'meta.json')) as f:
            meta = json.load(f)
        problem = meta['problem']
        result = {'problem': problem}
        if meta['szs_status'] in ['THM', 'CAX', 'UNS']:
            # We only care about successful refutation proofs.
            stdout_path = os.path.join(path, 'stdout.txt')
            signature = get_signature(problem)
            result['signature'] = signature
            # Assert that every symbol name is unique.
            assert len(signature) == len(set(signature))
            try:
                # Raises `RuntimeError` if the output file is too large.
                # Raises `ValueError` if no proof is found in the output file.
                result['clauses'] = load_proof_samples(stdout_path, signature, max_size)
            except (RuntimeError, ValueError) as e:
                log.warning(f'{stdout_path}: Failed to load proof: {str(e)}')
        return result

    print(f'Loading {len(paths)} proofs', file=sys.stderr)
    return parallel(joblib.delayed(load_one)(path) for path in paths)


def load_proof_samples(stdout_path, signature, max_size=None):
    if max_size is not None:
        cur_size = os.path.getsize(stdout_path)
        if cur_size > max_size:
            raise RuntimeError(f'Proof file is too large: {cur_size} > {max_size}')
    with open(stdout_path) as f:
        stdout = f.read()
    # Raises `ValueError` if no proof is found.
    df_formulas, df_operations = vampire.formulas.extract_df(stdout, roles=['proof', 'active'])
    df_formulas_active = df_formulas[df_formulas.role_active.notna()]
    log.debug('Clause count:\n%s' % yaml.dump({
        'total': len(df_formulas),
        'active': {
            'total': len(df_formulas_active),
            'proof': df_formulas_active.role_proof.count(),
            '~proof': df_formulas_active.role_proof.isna().sum()
        },
        'proof': df_formulas.role_proof.count()
    }, sort_keys=False))
    samples_list = list(df_to_samples(df_formulas_active, signature))
    if len(samples_list) == 0:
        raise RuntimeError(f'{stdout_path}: No proof samples were extracted.')
    samples_aggregated = {
        'token_counts': scipy.sparse.vstack(s['token_counts'] for s in samples_list),
        'proof': scipy.sparse.csc_matrix([[s['proof']] for s in samples_list], dtype=bool),
        'goal': scipy.sparse.csc_matrix([[s['goal']] for s in samples_list], dtype=bool),
    }
    return samples_aggregated


def df_to_samples(df, signature):
    for index, row in df.loc[:, ['string', 'role_proof', 'extra_goal']].iterrows():
        try:
            yield {
                'token_counts': clause_feature_vector(row.string, signature),
                'proof': pd.notna(row.role_proof),
                'goal': row.extra_goal
            }
        except RuntimeError as e:
            log.debug(str(e))
            continue


def clause_feature_vector(formula, signature):
    # Raises `RuntimeError` if parsing of the formula fails.
    try:
        # Raises `pyparsing.ParseException` if parsing of the formula fails.
        # Raises `RecursionError` if the formula is too deep.
        token_counts = vampire.clause.token_counts(formula)
    except (pyparsing.ParseException, RecursionError) as e:
        raise RuntimeError(f'{formula}: Failed to parse.') from e
    return token_counts_to_feature_vector(token_counts, signature)


def token_counts_to_feature_vector(token_counts, signature, dtype=np.uint32):
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
    signature = signature.tolist()
    indices += [4 + signature.index(s) for s in token_counts['symbol'].keys()]
    assert len(data) == len(indices)
    indptr = [0, len(data)]
    result = scipy.sparse.csr_matrix((data, indices, indptr), shape=(1, 4 + len(signature)), dtype=dtype)
    return result
