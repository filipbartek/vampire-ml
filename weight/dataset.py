import itertools
import logging
from collections import Counter
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
    X = clauses[df.index.get_level_values(False)] - clauses[df.index.get_level_values(True)]
    result = {'X': X, 'sample_weight': df}
    result['proof_search'] = {
        k: {
            'clauses': {
                False: sum(ps[False].values()),
                True: sum(ps[True].values())
            },
            'feature_vectors': {
                False: len(ps[False]),
                True: len(ps[True])
            }
        } for k, ps in proof_searches.items()
    }
    return result


def get_pair_indices(proof_searches, n_clauses, join_searches=False):
    proof_feature_vectors = set(itertools.chain.from_iterable(ps[True].keys() for ps in proof_searches.values()))

    def correct(proof_search):
        misclassified = Counter({k: v for k, v in proof_search[False].items() if k in proof_feature_vectors})
        if len(misclassified) > 0:
            return {
                False: proof_search[False] - misclassified,
                True: proof_search[True] + misclassified
            }
        return proof_search

    if join_searches:
        clause_weights = np.zeros((len(roles), n_clauses))
        for proof_search in proof_searches.values():
            proof_search = correct(proof_search)
            for (i, role) in enumerate(roles):
                counts = proof_search[role]
                n = sum(counts.values())
                clause_weights[i, list(counts.keys())] += np.fromiter(counts.values(), float, len(counts)) / n
        cw_dict = {role: dict(zip((cw > 0).nonzero()[0], cw[cw > 0])) for role, cw in zip(roles, clause_weights)}
        weight = get_pair_indices_one(cw_dict, name='all')
        df = weight.to_frame()
    else:
        pairs_hit = scipy.sparse.lil_matrix((n_clauses, n_clauses), dtype=bool)
        proof_search_weights = {}
        for i, proof_search in proof_searches.items():
            proof_search = correct(proof_search)
            with suppress(NoClausePairsError):
                weight = get_pair_indices_one(proof_search, name=i)
                assert np.isclose(1, weight.sum(), rtol=0, atol=1e-3)
                proof_search_weights[i] = weight
                assert np.all(weight > 0)
                pairs_hit[weight.index.get_level_values(False), weight.index.get_level_values(True)] = True
        if len(proof_search_weights) == 0:
            raise NoClausePairsError()
        pair_indices = pairs_hit.nonzero()
        index = pd.MultiIndex.from_arrays(pair_indices, names=[False, True])
        index.name = 'proof'
        proof_search_weights_aligned = {k: df.reindex(index, fill_value=0) for k, df in proof_search_weights.items()}
        df = pd.DataFrame(proof_search_weights_aligned)
    assert np.allclose(1, df.sum(), rtol=0, atol=1e-3)
    # `df`: Rows: clause pairs. Columns: proof searches. Cell values: weights. Each column is normalized (sums to 1).
    return df


def get_pair_indices_one(clause_weights, normalize=True, dtype_index=np.uint32, dtype_weight=np.float32, **kwargs):
    role_clauses = [np.fromiter(clause_weights[role].keys(), dtype_index, len(clause_weights[role])) for role in roles]
    pair_indices = np.meshgrid(*role_clauses, indexing='ij')
    pair_weights = np.outer(*(np.fromiter(clause_weights[role].values(), dtype=dtype_weight) for role in roles))
    if pair_weights.sum() == 0:
        raise NoClausePairsError()
    if normalize:
        pair_weights = pair_weights / pair_weights.sum()
    pair_indices_flat = [v.flatten() for v in pair_indices]
    pair_weights_flat = pair_weights.flatten()
    res = pd.Series(pair_weights_flat, index=pd.MultiIndex.from_arrays(pair_indices_flat, names=[False, True]), **kwargs)
    res.index.name = 'proof'
    return res
