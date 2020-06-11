import os


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
    try:
        return os.path.join(problems_path(), problem)
    except TypeError:
        return problem
