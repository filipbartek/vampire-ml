import itertools
import re
from collections import namedtuple

import pandas as pd

from utils import astype
from utils import invert_dict_of_lists

role_to_operations = {
    'input': [('PP', 'input')],
    'new': [('SA', 'new')],
    'passive': [('SA', 'passive')],
    'active': [('SA', 'active')]
}
operation_to_role = invert_dict_of_lists(role_to_operations)
formula_pattern = r'(?P<formula_id>\d+)\. (?P<formula>.+) \[(?P<inference_rule>[\w ]*)(?: (?P<inference_parents>[\d,]+))?\](?: \{(?P<extra>[\w,:-]*)\})?'
operation_pattern = fr'\[(?P<phase>\w+)\] (?P<operation>[\w ]+): {formula_pattern}'
FormulaOperation = namedtuple('FormulaOperation', ['id', 'formula', 'role', 'extra'])
supported_roles = ['proof'] + list(role_to_operations)


def extract_df(output, roles=None):
    # Raises `ValueError` if no proof is found.
    formulas = extract(output, roles=roles)
    records = [formula_to_record(k, v, roles=roles) for k, v in formulas.items()]
    df = pd.json_normalize(records, sep='_')
    df.set_index('formula_id', inplace=True)
    df.sort_index(inplace=True)
    if 'extra_goal' in df:
        df.extra_goal.fillna(0, inplace=True)
    dtype = {
        'extra_a': pd.UInt64Dtype(),
        'extra_w': pd.UInt64Dtype(),
        'extra_nSel': pd.UInt8Dtype(),
        'extra_thAx': pd.UInt64Dtype(),
        'extra_allAx': pd.UInt64Dtype(),
        'extra_thDist': pd.Int64Dtype(),
        'extra_goal': bool
    }
    df = astype(df, dtype, copy=False)
    return df


def formula_to_record(formula_id, properties, roles=None):
    if roles is None:
        roles = supported_roles
    result = {'formula_id': formula_id, 'formula': properties['formula'],
              'role': {role: role in properties['role'] for role in roles},
              'extra': properties['extra']}
    if 'goal' not in result['extra']:
        result['extra']['goal'] = None
    return result


def extract(output, roles=None):
    # Raises `ValueError` if no proof is found.
    proof_str = extract_proof(output)
    formula_operation_generators = []
    if roles is None or 'proof' in roles:
        formula_operation_generators.append(extract_operations(fr'^{formula_pattern}$', proof_str, default_role='proof'))
    roles_nonproof = None
    if roles is not None:
        roles_nonproof = [r for r in roles if r != 'proof']
    if roles_nonproof is None or len(roles_nonproof) >= 1:
        formula_operation_generators.append(extract_operations(fr'^{operation_pattern}$', output, roles=roles_nonproof))
    formula_operations = itertools.chain(*formula_operation_generators)
    #formula_operations = list(formula_operations)
    formulas = {}
    for res in formula_operations:
        assert res.role is not None
        if res.id not in formulas:
            formulas[res.id] = {
                'formula': res.formula,
                'role': set(),
                'extra': res.extra
            }
        extra_expected = formulas[res.id]['extra']
        if res.extra is not None:
            for k in res.extra:
                if k in extra_expected:
                    assert res.extra[k] == extra_expected[k]
                else:
                    formulas[res.id]['extra'][k] = res.extra[k]
        formulas[res.id]['role'].add(res.role)
    return formulas


def extract_proof(output):
    # Raises `ValueError` if no proof is found.
    parts = re.split(r'^% SZS output start Proof for (?P<problem>.+)$', output, maxsplit=1, flags=re.MULTILINE)
    if len(parts) != 3:
        # This happens namely when the problem is solved as satisfiable by saturating the passive clause set.
        raise ValueError('The output string does not contain a proof start line.')
    suffix = parts[2].lstrip()
    parts_2 = re.split(r'^% SZS output end Proof for (?P<problem>.+)$', suffix, maxsplit=1, flags=re.MULTILINE)
    if len(parts_2) != 3:
        raise ValueError('The output string does not contain a proof end line.')
    proof = parts_2[0].rstrip()
    return proof


def extract_operations(pattern, string, default_role=None, roles=None):
    for m in re.finditer(pattern, string, re.MULTILINE):
        try:
            role = operation_to_role[m['phase'], m['operation']]
        except (IndexError, KeyError):
            role = default_role
        if role is None:
            continue
        if roles is not None and role not in roles:
            continue
        formula_id = int(m['formula_id'])
        extra_dict = {}
        extra = m['extra']
        if extra is not None:
            pair_strings = (p.strip() for p in extra.split(','))
            string_pairs = (p.split(':', 1) for p in pair_strings)
            extra_dict = {k: int(v) for k, v in string_pairs}
        yield FormulaOperation(formula_id, m['formula'], role, extra_dict)
