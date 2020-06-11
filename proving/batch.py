import logging

import numpy as np

log = logging.getLogger(__name__)
dtype_embedding = np.float
symbol_embedding_column_names = ['arity', 'usageCnt', 'unitUsageCnt', 'inGoal', 'inUnit', 'introduced']


def generate_batch(problem_preference_matrices, batch_size, rng):
    problem_preference_matrices = list(problem_preference_matrices.items())
    problem_indexes = rng.choice(len(problem_preference_matrices), size=batch_size)
    symbol_pair_embeddings = []
    target_preference_values = []
    for problem_i, n_samples in zip(*np.unique(problem_indexes, return_counts=True)):
        problem = problem_preference_matrices[problem_i][0]
        preference_matrix = problem_preference_matrices[problem_i][1]['preference_matrix']
        symbols = problem_preference_matrices[problem_i][1]['symbols']
        try:
            symbol_pair_embedding, target_preference_value = generate_batch_from_one(symbols, preference_matrix,
                                                                                     n_samples, rng)
            symbol_pair_embeddings.append(symbol_pair_embedding)
            target_preference_values.append(target_preference_value)
        except RuntimeError:
            log.debug(f'Failed to generate samples from problem {problem}.', exc_info=True)
    # Throws ValueError if the contatenand is empty.
    return np.concatenate(symbol_pair_embeddings), np.concatenate(target_preference_values)


def generate_batch_from_one(symbols, preference_matrix, n_samples, rng, weighted_symbol_pairs=True):
    n = len(symbols)
    l, r = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    all_pairs_index_pairs = np.concatenate((l.reshape(-1, 1), r.reshape(-1, 1)), axis=1)
    all_pairs_values = preference_matrix.flatten()
    p = None
    if weighted_symbol_pairs and not np.allclose(0, all_pairs_values):
        p = np.abs(all_pairs_values) / np.sum(np.abs(all_pairs_values))
    chosen_pairs_indexes = rng.choice(len(all_pairs_index_pairs), size=n_samples, p=p)
    chosen_pairs_index_pairs = all_pairs_index_pairs[chosen_pairs_indexes]
    chosen_pairs_embeddings = get_symbol_pair_embeddings(symbols, chosen_pairs_index_pairs)
    chosen_pairs_values = all_pairs_values[chosen_pairs_indexes]
    return chosen_pairs_embeddings, chosen_pairs_values


def get_symbol_pair_embeddings(symbols, symbol_indexes):
    n_samples = len(symbol_indexes)
    assert symbol_indexes.shape == (n_samples, 2)
    symbol_embeddings = get_symbols_embedding(symbols, symbol_indexes.flatten()).reshape(n_samples, 2, -1)
    return np.concatenate((symbol_embeddings[:, 0], symbol_embeddings[:, 1]), axis=1)


def get_symbols_embedding(symbols, symbol_indexes):
    return get_all_symbol_embeddings(symbols)[symbol_indexes]


def get_all_symbol_embeddings(symbols):
    return symbols[symbol_embedding_column_names].to_numpy(dtype=dtype_embedding)
