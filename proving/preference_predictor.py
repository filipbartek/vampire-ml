import logging

import numpy as np

log = logging.getLogger(__name__)


# TODO: Simply implement batch generator that generates batches not from preference matrices, but rather from normalized Vampire measurements.
# Use that generator for scoring.
# Consider syncing the normalization between training and testing, possibly caching the fitted normalizer.


def evaluate_on_problem(problem, precedence_cost_generator, predictor):
    log.info('Evaluating on %s', problem)
    preference_matrices = {}
    symbol_type_records = []
    problem_record = {'problem': problem}
    try:
        # Throws RuntimeError when clausification fails.
        precedence_dicts, costs_normalized, symbols_by_type, rec = precedence_cost_generator.costs_of_random_precedences(
            problem)
        problem_record.update(rec)
        for symbol_type in precedence_cost_generator.random_precedence_types:
            symbol_type_record = {'problem': problem, 'symbol_type': symbol_type}
            precedences = np.asarray([precedence_dict[symbol_type] for precedence_dict in precedence_dicts])



            preference_matrix, row = fit_on_precedences(precedences, costs_normalized,
                                                        sklearn.base.clone(predictor_template), cross_validate)
            assert symbol_type not in preference_matrices
            if preference_matrix is not None:
                preference_matrices[symbol_type] = {'preference_matrix': preference_matrix,
                                                    'symbols': symbols_by_type[symbol_type]}
            symbol_type_record.update(row)
            symbol_type_records.append(symbol_type_record)
            # TODO: Cripple the predictor by aliasing by symbol embeddings.
            # TODO: Train a predictor on embeddings on one problem and symbol_type.
    except RuntimeError:
        log.debug('%s: Failed to estimate preference matrix.', problem, exc_info=True)
        problem_record['error'] = 'RuntimeError: Failed to estimate preference matrix.'
    return preference_matrices, problem_record, symbol_type_records
