import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import scipy
import yaml
from attributedict.collections import AttributeDict

from questions.memory import memory
from questions.utils import timer
from utils import is_compatible
from utils import subsample
from weight import vampire

log = logging.getLogger(__name__)


@memory.cache(ignore=['parallel'], verbose=2)
def load_proofs(paths, clausifier, clause_features, cfg, parallel=None, ss=None):
    def get_signature(problem):
        try:
            # Raises `RuntimeError` when clausification fails
            return clausifier.signature(problem)
        except RuntimeError:
            return None

    def load_one(path, seed):
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
                # Raises `RuntimeError` if the signature is too large.
                # Raises `RuntimeError` if the proof contains no nonproof clauses.
                # Raises `ValueError` if no proof is found in the output file.
                # Raises `ValueError` if a symbol encountered in a clause is missing from the signature.
                result['clauses'] = load_proof_samples(stdout_path, signature, clause_features, cfg, seed)
            except (RuntimeError, ValueError) as e:
                log.warning(f'{stdout_path}: Failed to load proof: {str(e)}')
        return result

    if ss is None:
        ss = np.random.SeedSequence(0)

    print(f'Loading {len(paths)} proofs', file=sys.stderr)
    return parallel(joblib.delayed(load_one)(path, seed) for path, seed in zip(paths, ss.spawn(len(paths))))


def load_proof_samples(stdout_path, signature, clause_features, cfg, seed):
    cfg = AttributeDict(cfg)
    if cfg.max_symbols is not None and len(signature) > cfg.max_symbols:
        raise RuntimeError(f'Signature is too large: {len(signature)} > {cfg.max_symbols}')
    if cfg.max_size is not None:
        actual_size = os.path.getsize(stdout_path)
        if actual_size > cfg.max_size:
            raise RuntimeError(f'Proof file is too large: {actual_size} > {cfg.max_size}')
    with open(stdout_path) as f:
        stdout = f.read()
    # Raises `ValueError` if no proof is found.
    df_formulas, df_operations = vampire.formulas.extract_df(stdout, roles=['proof', 'active'])
    df_samples = df_formulas.loc[df_formulas.role_active.notna(), ['string', 'role_proof', 'extra_goal']]
    df_samples = pd.concat([df_samples.loc[:, ['string', 'extra_goal']], df_samples.role_proof.notna()], axis='columns')

    category_indices = {
        'proof': df_samples.role_proof.to_numpy().nonzero()[0],
        'nonproof': np.logical_not(df_samples.role_proof.to_numpy()).nonzero()[0]
    }
    if len(category_indices['nonproof']) == 0:
        raise RuntimeError('The proof contains no active nonproof clauses.')

    rng = np.random.default_rng(seed)
    category_indices_selected = {k: subsample(v, cfg.max_clauses[k], rng) for k, v in category_indices.items()}
    all_selected_indices = np.concatenate(list(category_indices_selected.values()))
    df_samples = df_samples.iloc[all_selected_indices]

    log.debug('Clause count:\n%s' % yaml.dump({
        'total': len(df_formulas),
        'proof': df_formulas.role_proof.count(),
        'active&selected': {
            'total': len(df_samples),
            'proof': df_samples.role_proof.sum(),
            '~proof': df_samples.role_proof.sum()
        }
    }, sort_keys=False))

    log.debug(f'{stdout_path}: Parsing {len(df_samples)} clauses...')
    with timer() as t:
        samples_list = df_to_samples(df_samples, signature, clause_features)
    log.debug(f'{stdout_path}: {len(df_samples)} clauses parsed in {t.elapsed} s.')
    if len(samples_list) == 0:
        raise RuntimeError(f'{stdout_path}: No proof samples were extracted.')
    samples_aggregated = {
        'token_counts': scipy.sparse.vstack(s['token_counts'] for s in samples_list),
        'proof': scipy.sparse.csc_matrix([[s['proof']] for s in samples_list], dtype=bool),
        'goal': scipy.sparse.csc_matrix([[s['goal']] for s in samples_list], dtype=bool),
    }
    return samples_aggregated


def df_to_samples(df_samples, signature, clause_features):
    def row_to_sample(row):
        return {
            'token_counts': clause_feature_vector(row.string, signature, features=clause_features),
            'proof': row.role_proof,
            'goal': row.extra_goal
        }

    parallel = joblib.Parallel()
    return parallel(joblib.delayed(row_to_sample)(row) for index, row in df_samples.iterrows())


def clause_feature_vector(formula, signature, **kwargs):
    # Raises `pyparsing.ParseException` if parsing of the formula fails.
    # Raises `RecursionError` if the formula is too deep.
    token_counts = vampire.clause.token_counts(formula)
    return token_counts_to_feature_vector(token_counts, signature, **kwargs)


def token_counts_to_feature_vector(token_counts, signature, features=None, dtype=np.uint32):
    assert 0 <= token_counts['not'] <= token_counts['literal']
    all_features_dict = {
        'literal_positive': token_counts['literal'] - token_counts['not'],
        'literal_negative': token_counts['not'],
        'equality': token_counts['equality'],
        'inequality': token_counts['inequality'],
        'variable_occurrence': sum(token_counts['variable']),
        'variable_count': len(token_counts['variable']),
        'number': token_counts['number']
    }
    common_features = [all_features_dict[k] for k in features]
    return construct_feature_vector(common_features, token_counts['symbol'], signature, dtype)


def construct_feature_vector(common_features, symbol_counts, signature, dtype):
    """
    :param common_features: List of common feature values
    :param symbol_counts: Dict of symbol counts
    :param signature: List of symbol names
    :return: Sparse matrix with 1 row
    """
    if any(d > 0 for d in common_features):
        data, indices = map(list, zip(*((d, i) for i, d in enumerate(common_features) if d != 0)))
    else:
        data, indices = [], []
    data += symbol_counts.values()
    indices += [len(common_features) + signature.index(s) for s in symbol_counts.keys()]
    assert len(data) == len(indices)
    indptr = [0, len(data)]
    assert is_compatible(data, dtype)
    shape = (1, len(common_features) + len(signature))
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=dtype)
