import functools
import logging
import warnings

from questions import config
from questions.memory import memory
from . import process


def call_once(problem, options=None, timeout=None, precedences=None, get_stdout=True, get_stderr=True):
    logging.debug(f'Running Twee. Problem: {problem}')
    args = ['twee', config.full_problem_path(problem)] + options
    if precedences is not None:
        if 'predicate' in precedences:
            warnings.warn('Predicate precedences are not supported by Twee.')
        if 'function' in precedences:
            args.extend(['--precedence', ','.join(precedences['function'])])
    return process.run(args, timeout=timeout, capture_stdout=get_stdout, capture_stderr=get_stderr)


class Twee:
    def __init__(self, options=None, timeout=None, clausifier=None):
        self.options = options
        self.timeout = timeout
        self.clausifier = clausifier

    def solve(self, problem, precedences=None, cache=True):
        return self.call(problem, precedences=precedences, cache=cache)

    def call(self, problem, options=None, precedences=None, get_stdout=True, cache=True):
        if options is None:
            options = self.options

        if precedences is not None:
            clausification_result = self.clausifier.clausify(problem, get_symbols=True, get_clauses=False)
            precedences = {k: clausification_result.symbols_of_type(k).name[v] for k, v in precedences.items()}

        args = [problem]
        kwargs = {'options': options, 'timeout': self.timeout, 'precedences': precedences, 'get_stdout': get_stdout}
        if cache:
            call = functools.partial(memory.cache(call_once).call_and_shelve, *args, **kwargs)
            for i in range(2):
                memo_result = call()
                try:
                    # Raises KeyError if the file 'output.pkl' does not exist.
                    result = memo_result.get()
                except KeyError as e:
                    raise RuntimeError(
                        f'Problem {problem}: Attempt {i}: Failed to get memoized result. Try to increase the recursion limit.') from e
                if result is not None:
                    if result.returncode in (0, None):
                        break
                    warnings.warn(f'Problem {problem}: Attempt {i}: Unsupported return code: {result.returncode}')
                memo_result.clear()
        else:
            result = call_once(*args, **kwargs)
        return result
