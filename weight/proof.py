import json
import math
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import scipy

from questions.memory import memory
from questions.solver import Solver
from questions.utils import timer
from utils import is_compatible
from utils import sparse_equal
from utils import subsample
from weight import vampire

log = logging.getLogger(__name__)


@memory.cache(ignore=['parallel'], verbose=2)
def load_proofs(paths, clausifier=None, clause_features=None, max_size=None, parallel=None):
    # `ss` generates seeds to subsample clauses.
    if clausifier is None:
        clausifier = Solver()
    if parallel is None:
        parallel = joblib.Parallel()

    def get_signature(problem):
        try:
            # Raises `RuntimeError` when clausification fails
            return clausifier.signature(problem).tolist()
        except RuntimeError:
            return None

    def load_one(path):
        log.debug(f'Loading proof: {path}')
        with open(os.path.join(path, 'meta.json')) as f:
            meta = json.load(f)
        result = {k: meta[k] for k in ['problem', 'szs_status', 'activations', 'passive'] if k in meta}
        if meta['szs_status'] in ['THM', 'CAX', 'UNS']:
            # We only care about successful refutation proofs.
            # Only such runs output the proof in the supported format (list of useful activated clauses).
            stdout_path = os.path.join(path, 'stdout.txt')
            actual_size = os.path.getsize(stdout_path)
            log.debug(f'{stdout_path}: Proof file size: {actual_size}')
            result['size'] = actual_size
            signature = get_signature(meta['problem'])
            result['signature'] = signature
            # Assert that every symbol name is unique.
            assert len(signature) == len(set(signature))
            with timer() as t:
                try:
                    if max_size is not None and actual_size > max_size:
                        raise RuntimeError(f'Proof file is too large: {actual_size} > {max_size}')
                    # Raises `RuntimeError` if the output file is too large.
                    # Raises `RuntimeError` if the signature is too large.
                    # Raises `RuntimeError` if the proof contains no nonproof clauses.
                    # Raises `ValueError` if no proof is found in the output file.
                    # Raises `ValueError` if a symbol encountered in a clause is missing from the signature.
                    result['clauses'] = load_proof_samples(stdout_path, signature, clause_features)
                except (RuntimeError, ValueError) as e:
                    log.warning(f'{stdout_path}: Failed to load proof: {str(e)}')
                    result['error'] = {'type': type(e).__name__, 'message': str(e)}
            result['time_load'] = t.elapsed
        return result

    print(f'Loading {len(paths)} proofs', file=sys.stderr)
    return parallel(joblib.delayed(load_one)(path) for path in paths)


def load_proof_samples(stdout_path, signature, clause_features):
    with open(stdout_path) as f:
        stdout = f.read()
    return stdout_to_proof_samples(stdout, signature, clause_features)


def stdout_to_proof_samples(stdout, signature, clause_features, clause_max_len=None, clause_max_terminals=None):
    # Raises `ValueError` if no proof is found.
    df_formulas, df_operations = vampire.formulas.extract_df(stdout, roles=['proof', 'active'])
    df_samples = df_formulas.loc[df_formulas.role_active.notna(), ['string', 'role_proof', 'extra_goal']]
    df_samples = pd.concat([df_samples.loc[:, ['string', 'extra_goal']], df_samples.role_proof.notna()], axis='columns')

    log.debug(f'Parsing {len(df_samples)} clauses...')
    feature_vectors = df_to_samples(df_samples, signature, clause_features, clause_max_len=clause_max_len,
                                    clause_max_terminals=clause_max_terminals)
    log.debug(f'{len(df_samples)} clauses parsed. Unique feature vectors: {len(feature_vectors)}.')
    return feature_vectors


def df_to_samples(df_samples, signature, clause_features, clause_max_len=None, clause_max_terminals=None,
                  parallel=None):
    if parallel is None:
        parallel = joblib.Parallel(verbose=1)

    def formula_to_feature_vector(formula):
        with timer() as t:
            try:
                # Note: We assume that failing to parse one clause makes all the remaining clauses useless.
                # For this reason we allow exception to propagate up from the parallel call.
                return clause_feature_vector(formula, signature, features=clause_features,
                                             max_terminals=clause_max_terminals)
            except vampire.clause.MaxTerminalsError as e:
                # We ignore clauses that are too large.
                log.debug(f'{e} Characters: {len(formula)}. Time: {t.elapsed}')
                return None

    if clause_max_len is not None:
        selected = df_samples.string.str.len() <= clause_max_len
        if selected.sum() < len(selected):
            log.debug(f'{(~selected).sum()}/{len(selected)} clauses are excluded because they have more than {clause_max_len} characters.')
            df_samples = df_samples[selected]

    print(f'Converting {len(df_samples)} clauses to feature vectors. Signature size: {len(signature)}. Maximum clause length: {df_samples.string.str.len().max()} characters.',
          file=sys.stderr)
    feature_vectors = parallel(joblib.delayed(formula_to_feature_vector)(formula) for formula in df_samples.string)

    result = {}
    for feature_vector, role_proof in zip(feature_vectors, df_samples.role_proof):
        if feature_vector is None:
            continue
        h = joblib.hash(feature_vector)
        if h not in result:
            result[h] = {
                'feature_vector': feature_vector,
                'role_proof': {
                    False: 0,
                    True: 0
                }
            }
        assert sparse_equal(result[h]['feature_vector'], feature_vector)
        result[h]['role_proof'][role_proof] += 1
    return result


def clause_feature_vector(formula, signature, max_terminals=None, **kwargs):
    # Raises `pyparsing.ParseException` if parsing of the formula fails.
    # Raises `RecursionError` if the formula is too deep.
    # Raises `ValueError` if a parse error occurs or if the formula is too large.
    token_counts = vampire.clause.token_counts(formula, max_terminals=max_terminals)
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
    if features is None:
        features = all_features_dict.keys()
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


def subsample_proof(token_counts, proof, max_sample_size, rng):
    if max_sample_size is not None:
        n_proof = proof.nnz
        n_nonproof = proof.shape[0] - n_proof
        n_features = token_counts.shape[1]
        if n_proof * n_nonproof * n_features > max_sample_size:
            max_clause_pairs = max_sample_size // n_features
            if max_clause_pairs == 0:
                raise ValueError(f'The signature is too large: {n_features} > {max_sample_size}.')
            max_per_type = int(math.sqrt(max_clause_pairs))
            assert max_per_type * max_per_type <= max_clause_pairs
            assert 1 <= max_per_type
            max_proof = min(max_per_type, n_proof)
            max_nonproof = min(max_per_type, n_nonproof)
            if n_proof <= max_per_type:
                assert max_proof == n_proof
                max_nonproof = min(max_clause_pairs // max_proof, n_nonproof)
            elif n_nonproof <= max_per_type:
                max_proof = min(max_clause_pairs // max_nonproof, n_proof)
            assert 1 <= max_proof <= n_proof
            assert 1 <= max_nonproof <= n_nonproof
            assert max_proof * max_nonproof <= max_clause_pairs
            proof_dense = proof.toarray().squeeze(1)
            assert len(proof_dense.shape) == 1
            clauses_proof = subsample(np.where(proof_dense)[0], max_proof, rng=rng)
            clauses_nonproof = subsample(np.where(~proof_dense)[0], max_nonproof, rng=rng)
            clauses_selected = np.concatenate([clauses_nonproof, clauses_proof])
            token_counts = token_counts[clauses_selected]
            proof = proof[clauses_selected]
            log.debug(
                f'Proof subsampled. Features: {n_features}. Original clauses: {n_proof}x{n_nonproof}. New clauses: {max_proof}x{max_nonproof}.')
    return token_counts, proof
