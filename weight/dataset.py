import itertools
import logging
from contextlib import suppress

import joblib
import numpy as np
import pandas as pd
import scipy

from utils import sparse_equal

log = logging.getLogger(__name__)
roles = [False, True]


class NoClausePairsError(ValueError):
    def __init__(self):
        super().__init__('No nonproof-proof clause pairs were found.')


def proofs_to_samples(feature_vectors, proof_searches, max_sample_size=None, rng=None, flip_odd=True, **kwargs):
    result = get_all_samples(feature_vectors, proof_searches, **kwargs)

    # By default, each pair is used once.
    n_samples, n_features = result['X'].shape
    if max_sample_size is not None and n_samples * n_features > max_sample_size:
        max_pairs = max_sample_size // n_features
        log.debug(f'Subsampling clause pairs. Before: {n_samples}. After: {max_pairs}.')
        # We sample with replacement to ensure that the final distribution respects `sample_weight`.
        new_indices = rng.choice(n_samples, size=max_pairs, p=result['sample_weight'])
        result['X'] = result['X'][new_indices]
        del result['sample_weight']
        n_samples = max_pairs

    y = np.ones(n_samples, dtype=bool)
    if flip_odd:
        # Flip odd samples
        y[1::2] = False
        result['X'] = result['X'].multiply(np.expand_dims(np.where(y, 1, -1), 1))
    result['y'] = y
    return result


def get_all_samples(feature_vectors, proof_searches, dtype=np.int32, **kwargs):
    n_clauses = len(feature_vectors)
    df = get_pair_indices(proof_searches, n_clauses, **kwargs)
    clauses = scipy.sparse.vstack(feature_vectors.values(), format='csr', dtype=dtype)

    samples = {}
    for i_false, i_true, weight in df.itertuples(index=False):
        # nonproof - proof
        feature_difference = clauses[i_false] - clauses[i_true]
        h = joblib.hash(feature_difference)
        if h not in samples:
            samples[h] = {
                'feature_difference': feature_difference,
                'weight': 0
            }
        assert sparse_equal(samples[h]['feature_difference'], feature_difference)
        samples[h]['weight'] += weight
    X = scipy.sparse.vstack(sample['feature_difference'] for sample in samples.values())
    sample_weight = np.fromiter((sample['weight'] for sample in samples.values()), float, len(samples))

    assert np.isclose(1, sample_weight.sum())
    return {'X': X, 'sample_weight': sample_weight}


def get_pair_indices(proof_searches, n_clauses, join_searches=False):
    if join_searches:
        clause_weights = np.zeros((len(roles), n_clauses))
        for (i, role), proof_search in itertools.product(enumerate(roles), proof_searches):
            counts = proof_search[role]
            n = sum(counts.values())
            clause_weights[i, list(counts.keys())] += np.fromiter(counts.values(), float, len(counts)) / n
        cw_dict = {role: dict(zip((cw > 0).nonzero()[0], cw[cw > 0])) for role, cw in zip(roles, clause_weights)}
        df = get_pair_indices_one(cw_dict)
    else:
        pair_weights = scipy.sparse.lil_matrix((n_clauses, n_clauses))
        for proof_search in proof_searches:
            with suppress(NoClausePairsError):
                df = get_pair_indices_one(proof_search)
                pair_weights[df[False], df[True]] += df.weight.to_numpy()
        if pair_weights.nnz == 0:
            raise NoClausePairsError()
        pair_indices = pair_weights.nonzero()
        data = {
            **{k: v for k, v in zip(roles, pair_indices)},
            'weight': pair_weights[pair_indices[0], pair_indices[1]].toarray()[0]
        }
        data['weight'] /= data['weight'].sum()
        df = pd.DataFrame(data)
    assert np.isclose(1, df.weight.sum())
    return df


def get_pair_indices_one(clause_weights):
    role_clauses = [np.fromiter(clause_weights[role].keys(), np.uint32, len(clause_weights[role])) for role in roles]
    pair_indices = np.meshgrid(*role_clauses, indexing='ij')
    pair_weights = np.outer(*(list(clause_weights[role].values()) for role in roles))
    if pair_weights.sum() == 0:
        raise NoClausePairsError()
    data = {
        **{k: v.flatten() for k, v in zip(roles, pair_indices)},
        'weight': pair_weights.flatten() / pair_weights.sum()
    }
    df = pd.DataFrame(data)
    return df
