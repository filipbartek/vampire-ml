import os
import re

import appdirs


def program_name():
    return 'vampire_ml'


def program_version():
    return '1.0'


def cache_dir():
    return appdirs.user_cache_dir(program_name(), version=program_version())


def scratch_dir():
    try:
        return os.environ['SCRATCH']
    except KeyError:
        pass
    return None


def tptp_path():
    try:
        return os.environ['TPTP']
    except KeyError:
        pass
    return None


def problems_path():
    try:
        return os.path.join(tptp_path(), 'Problems')
    except TypeError:
        return None


def full_problem_path(problem, dir_paths=None):
    if dir_paths is None:
        dir_paths = []
    m = re.search(
        r'\b(?P<problem_name>(?P<problem_domain>[A-Z]{3})(?P<problem_number>[0-9]{3})(?P<problem_form>[-+^=_])(?P<problem_version>[1-9])(?P<problem_size_parameters>[0-9]*(\.[0-9]{3})*))\b',
        problem)
    if m is not None and problem == m['problem_name']:
        problem = os.path.join(m['problem_domain'], f'{problem}.p')
        dir_paths = dir_paths + [problems_path()]
    for dir_path in ['.'] + dir_paths:
        result = os.path.join(dir_path, problem)
        if os.path.isfile(result):
            return os.path.normpath(result)
    raise RuntimeError(f'Cannot find problem: {problem}')
