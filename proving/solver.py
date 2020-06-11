import copy
import logging

from joblib import Parallel, delayed

from proving import vampire

log = logging.getLogger(__name__)


class Solver:
    default_options = {
        'encode': 'on',
        'statistics': 'full',
        'time_statistics': 'on',
        'proof': 'off',
        'literal_comparison_mode': 'predicate',
        'symbol_precedence': 'frequency',
        'saturation_algorithm': 'discount',
        'age_weight_ratio': '10',
        'avatar': 'off',
        'time_limit': '10'
    }

    def __init__(self, options=None, timeout=None):
        if options is None:
            options = self.default_options
        self.options = options
        self.timeout = timeout

    def __repr__(self):
        return f'{type(self).__name__}(options={self.options}, timeout={self.timeout})'

    def costs_of_random_precedences(self, problem, random_precedence_types, seeds):
        # Throws RuntimeError when clausification fails.
        symbols_by_type = {symbol_type: self.symbols_of_type(problem, symbol_type) for symbol_type in
                           random_precedence_types}
        symbol_counts = {symbol_type: len(symbols) for symbol_type, symbols in symbols_by_type.items()}
        return Parallel()(
            delayed(self._cost_of_one_randomized_solution)(problem, symbol_counts, seed) for seed in
            seeds), symbols_by_type

    def _cost_of_one_randomized_solution(self, problem, symbol_counts, seed):
        precedences = {symbol_type: self.random_precedence(length, symbol_type, seed) for symbol_type, length in
                       symbol_counts.items()}
        result = self.solve(problem, precedences)
        if result.returncode == 0:
            cost = result.saturation_iterations
        else:
            cost = None
        return precedences, cost

    def random_precedence(self, length, symbol_type, seed):
        return vampire.random_precedence(symbol_type=symbol_type, length=length, seed=seed)

    def symbols_of_type(self, problem, symbol_type):
        result = self.clausify(problem, get_clauses=False)
        if result.returncode != 0:
            raise RuntimeError('Clausification failed.')
        return result.symbols_of_type(symbol_type)

    def get_symbol_count(self, problem, symbol_type):
        return len(self.symbols_of_type(problem, symbol_type))

    def clausify(self, problem, get_symbols=True, get_clauses=True, get_stdout=False):
        options = copy.deepcopy(self.options)
        if options is None:
            options = {}
        if 'mode' in options:
            log.warning('Overriding mode.')
        options['mode'] = 'clausify'
        return self.call(problem, options=options, get_symbols=get_symbols, get_clauses=get_clauses,
                         get_stdout=get_stdout)

    def solve(self, problem, precedences=None):
        return self.call(problem, precedences=precedences)

    def call(self, problem, options=None, precedences=None, get_symbols=False, get_clauses=False, get_stdout=True):
        if options is None:
            options = self.options
        return vampire.call(problem, options=options, timeout=self.timeout, precedences=precedences,
                            get_symbols=get_symbols, get_clauses=get_clauses, get_stdout=get_stdout)
