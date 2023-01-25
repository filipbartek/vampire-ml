import itertools
import logging
from contextlib import suppress

import numpy as np
import pandas as pd
import scipy

log = logging.getLogger(__name__)
roles = [False, True]


class NoClausePairsError(ValueError):
    def __init__(self):
        super().__init__('No nonproof-proof clause pairs were found.')


def proofs_to_samples(feature_vectors, proof_searches, **kwargs):
    return get_all_samples(feature_vectors, proof_searches, **kwargs)


def get_all_samples(feature_vectors, proof_searches, dtype=np.int32, **kwargs):
    n_clauses = len(feature_vectors)
    try:
        df = get_pair_indices(proof_searches, n_clauses, **kwargs)
    except NoClausePairsError:
        n_features = 0
        if len(feature_vectors) >= 1:
            n_features = feature_vectors[0].shape[1]
        return {'X': scipy.sparse.csr_matrix((0, n_features), dtype=dtype)}
    clauses = scipy.sparse.vstack(feature_vectors, format='csr', dtype=dtype)

    X = clauses[df[False]] - clauses[df[True]]
    sample_weight = df.weight

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
