import os
import re


def program_name():
    return 'vampire_ml'


def program_version():
    return '1.0'


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


def full_problem_path(problem):
    m = re.search(
        r'\b(?P<problem_name>(?P<problem_domain>[A-Z]{3})(?P<problem_number>[0-9]{3})(?P<problem_form>[-+^=_])(?P<problem_version>[1-9])(?P<problem_size_parameters>[0-9]*(\.[0-9]{3})*))\b',
        problem)
    if m is not None and problem == m['problem_name']:
        problem = os.path.join(m['problem_domain'], f'{problem}.p')
    try:
        return os.path.join(problems_path(), problem)
    except TypeError:
        return problem
