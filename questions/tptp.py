import itertools
import os
import re
import warnings

import pandas as pd

from questions import config
from questions import utils

property_types = {
    'domain': 'category',  # Domain name abbreviation. 3 letters
    'number': pd.UInt16Dtype(),  # Abstract problem number. 3 digits - 0 to 999. Unique within the domain.
    'form': 'category',  # ^ for THF, _ for TFF without arithmetic, = for TFF with arithmetic, + for FOF, and - for CNF.
    'version': pd.UInt8Dtype(),
    'source': 'category',
    'status': 'category',
    'rating': float,
    'spc': 'category',
    'formulae': pd.UInt32Dtype(),
    'formulae_unit': pd.UInt32Dtype(),
    'formulae_def': pd.UInt32Dtype(),
    'clauses': pd.UInt32Dtype(),
    'atoms': pd.UInt32Dtype(),
    'atoms_equality': pd.UInt32Dtype(),
    'atoms_neg': pd.UInt32Dtype(),
    'clause_size_max': pd.UInt16Dtype(),
    'clause_size_avg': pd.UInt16Dtype(),
    'formula_depth_max': pd.UInt32Dtype(),
    'formula_depth_avg': pd.UInt32Dtype(),
    'predicates': pd.UInt32Dtype(),
    'predicates_user': pd.UInt32Dtype(),
    'predicates_propositional': pd.UInt32Dtype(),
    'predicates_arity_min': pd.UInt16Dtype(),
    'predicates_arity_max': pd.UInt16Dtype(),
    'functors': pd.UInt32Dtype(),
    'functors_user': pd.UInt32Dtype(),
    'functors_constant': pd.UInt32Dtype(),
    'variables': pd.UInt32Dtype(),
    'variables_singleton': pd.UInt32Dtype(),
    'variables_universal': pd.UInt32Dtype(),
    'variables_existential': pd.UInt32Dtype(),
    'term_depth_max': pd.UInt16Dtype(),
    'term_depth_avg': pd.UInt16Dtype()
}

# http://www.tptp.org/TPTP/TR/TPTPTR.shtml#HeaderSection
header_patterns = {
    r'^% Domain +: (?P<domain_full>.*)$': None,
    r'^% Source +: (?P<source>.*)$': None,
    r'^% Status +: (?P<status>.*)$': None,
    r'^% Rating +: (?P<rating>(\d\.\d\d|\?)) (?P<rating_release>v\d\.\d\.\d)\b': None,
    r'^% SPC +: (?P<spc>.*)$': None,
    r'^% .*\bNumber of clauses +: +(?P<clauses>\d+)\b': {'-'},
    r'^% .*\bNumber of formulae +: +(?P<formulae>\d+) \( *(?P<formulae_unit>\d+) (?:unit|unt)(?:; *(?P<formulae_def>\d+) def)?\)$': {'+'},
    r'^% .*\bNumber of (?:atoms|literals) +: +(?P<atoms>\d+) \( *(?P<atoms_equality>\d+) (?:equality|equ)(?:; *(?P<atoms_neg>\d+) neg)?\)$': {'-', '+'},
    r'^% .*\bMaximal clause size +: +(?P<clause_size_max>\d+) \( *(?P<clause_size_avg>\d+) (?:average|avg)\)$': {'-'},
    r'^% .*\bMaximal formula depth +: +(?P<formula_depth_max>\d+) \( *(?P<formula_depth_avg>\d+) (?:average|avg)\)$': {'+'},
    r'^% .*\bNumber of predicates +: +(?P<predicates>\d+) \((?: *(?P<predicates_user>\d+) usr;)? *(?P<predicates_propositional>\d+) (?:propositional|prp); *(?P<predicates_arity_min>\d+)-(?P<predicates_arity_max>\d+) (?:arity|aty)\)$': {'-', '+'},
    r'^% .*\bNumber of functors +: +(?P<functors>\d+) \((?: *(?P<functors_user>\d+) usr;)? *(?P<functors_constant>\d+) (?:constant|con).*\)$': {'-', '+'},
    r'^% .*\bNumber of variables +: +(?P<variables>\d+) \( *(?P<variables_singleton>\d+) (?:singleton|sgn)\)$': {'-'},
    # https://stackoverflow.com/a/6576808/4054250
    r'^% .*\bNumber of variables +: +(?P<variables>\d+) \((?: *(?P<variables_singleton>\d+) (?:singleton|sgn);?)? *(?P<variables_universal>\d+) +!;? *(?P<variables_existential>\d+) +\?\)$': {'+'},
    r'^% .*\bMaximal term depth +: +(?P<term_depth_max>\d+) \( *(?P<term_depth_avg>\d+) (?:average|avg)\)$': {'-', '+'}
}


def problem_properties(problem, header_properties=True):
    res = file_name_properties(problem)
    if header_properties:
        try:
            form = res['form']
        except KeyError:
            form = None
        res.update(problem_header_properties(header(problem), form))
    return res


def file_name_properties(file_name):
    file_name = os.path.basename(file_name)
    # http://www.tptp.org/TPTP/TR/TPTPTR.shtml#ProblemAndAxiomatizationNaming
    p = r'^(?P<name>(?P<domain>[A-Z]{3})(?P<number>[0-9]{3})(?P<form>[-+^=_])(?P<version>[1-9])(?P<size_parameters>[0-9]*(\.[0-9]{3})*))(?:\.[pg])?$'
    return match_and_cast(p, file_name)


def problem_header_properties(content, form=None):
    return utils.join_dicts(match_and_cast(p, content) for p, forms in header_patterns.items() if forms is None or form in forms)


def match_and_cast(pattern, string):
    m = re.search(pattern, string, re.MULTILINE)
    if m is None:
        return {}
    return {name: cast_value(name, string_value) for name, string_value in m.groupdict().items() if string_value is not None}


def cast_value(name, raw_value):
    if raw_value is None:
        raise ValueError(f'Attempting to cast value None of parameter {name}.')
    try:
        property_type = property_types[name]
        if property_type in ('object', 'category'):
            return raw_value
        if isinstance(property_type, pd.api.extensions.ExtensionDtype):
            return property_type.type(raw_value)
        return property_type(raw_value)
    except KeyError:
        return raw_value
    except ValueError:
        return None


def header(problem):
    file_path = config.full_problem_path(problem)
    with open(file_path) as f:
        header_lines = itertools.takewhile(lambda l: l.startswith('%') or not l.rstrip(), f)
        return ''.join(header_lines)
