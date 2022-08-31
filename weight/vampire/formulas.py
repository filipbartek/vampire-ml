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
roles = ['proof'] + list(role_to_operations)


def extract_df(output):
    formulas = extract(output)
    records = [formula_to_record(k, v) for k, v in formulas.items()]
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


def formula_to_record(formula_id, properties):
    result = {'formula_id': formula_id, 'formula': properties['formula'],
              'role': {role: role in properties['role'] for role in roles},
              'extra': properties['extra']}
    if 'goal' not in result['extra']:
        result['extra']['goal'] = None
    return result


def extract(output):
    proof_str = extract_proof(output)
    formula_operations = itertools.chain(extract_operations(fr'^{formula_pattern}$', proof_str, 'proof'),
                                         extract_operations(fr'^{operation_pattern}$', output))
    formula_operations = list(formula_operations)
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
    parts = re.split(r'^% SZS output start Proof for (?P<problem>.+)$', output, maxsplit=1, flags=re.MULTILINE)
    assert len(parts) == 3
    suffix = parts[2].lstrip()
    parts_2 = re.split(r'^% SZS output end Proof for (?P<problem>.+)$', suffix, maxsplit=1, flags=re.MULTILINE)
    assert len(parts_2) == 3
    proof = parts_2[0].rstrip()
    return proof


def extract_operations(pattern, string, default_role=None):
    for m in re.finditer(pattern, string, re.MULTILINE):
        try:
            role = operation_to_role[m['phase'], m['operation']]
        except (IndexError, KeyError):
            role = default_role
        if role is None:
            continue
        formula_id = int(m['formula_id'])
        extra_dict = {}
        extra = m['extra']
        if extra is not None:
            pair_strings = (p.strip() for p in extra.split(','))
            string_pairs = (p.split(':', 1) for p in pair_strings)
            extra_dict = {k: int(v) for k, v in string_pairs}
        yield FormulaOperation(formula_id, m['formula'], role, extra_dict)
