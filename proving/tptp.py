import os
import re

import config


def problem_properties(problem):
    res = problem_name_properties(problem)
    res.update(problem_header_properties(open(config.full_problem_path(problem)).read()))
    return res


def problem_name_properties(problem):
    # http://www.tptp.org/TPTP/TR/TPTPTR.shtml#ProblemAndAxiomatizationNaming
    m = re.search(
        r'^(?P<domain>[A-Z]{3})(?P<number>[0-9]{3})(?P<form>[-+^=_])(?P<version>[1-9])(?P<size_parameters>[0-9]*(\.[0-9]{3})*)\.p$',
        os.path.basename(problem))
    if m is None:
        return {}
    return {'domain': m['domain'],
            'number': int(m['number']),
            'form': m['form'],
            'version': int(m['version']),
            'size_parameters': m['size_parameters']}


def problem_header_properties(content):
    # http://www.tptp.org/TPTP/TR/TPTPTR.shtml#HeaderSection
    res = {
        'domain_full': re_search(r'^% Domain   : (.*)$', content),
        'source': re_search(r'^% Source   : (.*)$', content),
        'status': re_search(r'^% Status   : (.*)$', content),
        'rating': re_search(r'^% Rating   : (\d\.\d\d) (v\d\.\d\.\d)\b', content, cast=float),
        'spc': re_search(r'^% SPC      : (.*)$', content)
    }
    res.update(atoms(content))
    return res


def atoms(content):
    m = re.search(r'^% +Number of atoms +: +(?P<atoms>\d+) \( *(?P<atoms_equality>\d+) equality\)$',
                  content, re.MULTILINE)
    if m is None:
        return {}
    res = {k: int(v) for k, v in m.groupdict().items()}
    res['equality_present'] = res['atoms_equality'] != 0
    return res


def re_search(pattern, string, cast=None):
    m = re.search(pattern, string, re.MULTILINE)
    if m is None:
        return None
    if cast is not None:
        return cast(m[1])
    return m[1]
