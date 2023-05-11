import os
import re
from contextlib import suppress

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
    file_paths = [os.path.join(p, problem) for p in dir_paths if p is not None] + [problem]
    m = re.search(
        r'(?P<path>.*/)?(?P<problem_name>(?P<problem_domain>[A-Z]{3})(?P<problem_number>[0-9]{3})(?P<problem_form>[-+^=_])(?P<problem_version>[1-9])(?P<problem_size_parameters>[0-9]*(\.[0-9]{3})*))(?P<extension>\.[pq])?',
        problem)
    if m is not None:
        with suppress(TypeError):
            file_paths.append(os.path.join(problems_path(), m['problem_domain'], m['problem_name'] + '.p'))
    for result in file_paths:
        if os.path.isfile(result):
            return os.path.normpath(result)
    raise RuntimeError(f'Cannot find problem: {problem}')
