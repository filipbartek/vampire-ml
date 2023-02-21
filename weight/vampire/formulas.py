import itertools
import re
import warnings
from contextlib import suppress

import numpy as np
import pandas as pd

from utils import astype
from utils import invert_dict_of_lists

role_to_operations = {
    'input': [('PP', 'input')],
    'new': [('SA', 'new')],
    'passive': [('SA', 'passive')],
    'active': [('SA', 'active')],
    'selected': [('SA', 'selected')]
}
operation_to_role = invert_dict_of_lists(role_to_operations)
supported_roles = ['proof'] + list(role_to_operations)


def extract_df(output, roles=None):
    # Raises `ValueError` if no proof is found.
    formulas, operations = extract(output, roles=roles)
    df_operations = pd.json_normalize(operations, sep='_')
    df_operations = astype(df_operations, {
        'formula_id': np.uint64,
        'role': 'category',
        'inference_rule': 'category',
        'span_start': np.uint64,
        'span_end': np.uint64
    }, copy=False)
    df_formulas = pd.json_normalize(formulas.values(), sep='_')
    df_formulas.set_index(pd.Index(formulas.keys(), dtype=np.uint32, name='id'), inplace=True)
    df_formulas.sort_index(inplace=True)
    if 'extra_goal' in df_formulas:
        df_formulas.extra_goal.fillna(0, inplace=True)
    else:
        df_formulas['extra_goal'] = False
    if roles is not None:
        for role in roles:
            col = f'role_{role}'
            if col not in df_formulas:
                df_formulas[col] = pd.NA
    dtype = {f'extra_{k}': v for k, v in {
        'a': pd.UInt64Dtype(),
        'w': pd.UInt64Dtype(),
        'nSel': pd.UInt8Dtype(),
        'thAx': pd.UInt64Dtype(),
        # Seems to be positive integer with one exception: INT_MIN.
        'allAx': pd.Int64Dtype(),
        # Seems to be negative integer with one exception: -inf.
        'thDist': float,
        'goal': bool,
        'selection_queue': pd.CategoricalDtype(['a', 'w'])
    }.items()}
    dtype.update({f'role_{role}': pd.UInt64Dtype() for role in roles})
    df_formulas = astype(df_formulas, dtype, copy=False)
    return df_formulas, df_operations


def extract(output, roles=None):
    formula_operation_generators = []
    # We extract proof formulas first to fail early in case the proof segment of `output` is missing or incomplete.
    if roles is None or 'proof' in roles:
        formula_operation_generators.append(extract_operations_proof(output))
    formula_operation_generators.append(
        extract_operations_nonproof(output, [r for r in roles if r not in ['proof', 'selected']]))
    if 'selected' in roles:
        formula_operation_generators.append(extract_operations(fr'^{selected_pattern}$', output, roles=['selected']))
    formulas = {}
    operations = []
    for op in itertools.chain(*formula_operation_generators):
        formula_id = op['formula']['id']
        if formula_id not in formulas:
            formulas[formula_id] = {k: v for k, v in op['formula'].items() if k != 'id'}
            formulas[formula_id]['role'] = {}
        extra = op['formula']['extra']
        if extra is not None:
            for k in extra:
                if k not in formulas[formula_id]['extra']:
                    formulas[formula_id]['extra'][k] = extra[k]
                expected = formulas[formula_id]['extra'][k]
                if not are_equal(extra[k], expected):
                    warnings.warn(f'Clause extra information mismatch: formula={formula_id}, key={k}, expected={expected}, actual={extra[k]}')
        role = op['role']
        if role in formulas[formula_id]['role']:
            raise RuntimeError(f'Formula {formula_id} encountered in role {role} more than once.')
        formulas[formula_id]['role'][role] = op['span']['start']
        operations.append({
            'formula_id': formula_id,
            **{k: v for k, v in op.items() if k != 'formula'}
        })
    return formulas, operations


def are_equal(a, b, equal_nan=True):
    if equal_nan:
        with suppress(TypeError):
            if np.isnan(a) and np.isnan(b):
                return True
    if a == b:
        return True
    return False


formula_pattern = r'(?P<formula_id>\d+)\. (?P<formula>.+) \[(?P<inference_rule>[a-z ]*)(?: (?P<inference_parents>[\d,]+))?\](?: \{(?P<extra>[\w,:\-\.]*)\})?'
operation_pattern = fr'\[(?P<phase>\w+)\] (?P<operation>[\w ]+): {formula_pattern}'
selected_pattern = r'\[(?P<phase>\w+)\] (?P<operation>[\w ]+): (?P<formula_id>\d+)\. \{(?P<extra>[\w,:\-\.]*)\}'


def extract_operations_nonproof(output, roles=None):
    roles_nonproof = None
    if roles is not None:
        roles_nonproof = [r for r in roles if r != 'proof']
    if roles_nonproof is None or len(roles_nonproof) >= 1:
        return extract_operations(fr'^{operation_pattern}$', output, roles=roles_nonproof)
    return []


def extract_operations_proof(output):
    with suppress(ValueError):
        return extract_operations(fr'^{formula_pattern}$', output, default_role='proof',
                                  pos=extract_proof_start(output), endpos=extract_proof_end(output))
    return []


def extract_proof_start(output):
    m = re.search(r'^% SZS output start Proof for (?P<problem>.+)$', output, flags=re.MULTILINE)
    if m is None:
        raise ValueError('Proof start not found.')
    return m.end() + 1


def extract_proof_end(output):
    m = re.search(r'^% SZS output end Proof for (?P<problem>.+)$', output, flags=re.MULTILINE)
    if m is None:
        raise ValueError('Proof end not found.')
    return m.start()


def extract_operations(pattern, string, default_role=None, roles=None, **kwargs):
    for m in re.compile(pattern, re.MULTILINE).finditer(string, **kwargs):
        try:
            # Phase and operation is only extracted if the pattern contains the respective named fields.
            # `operation_pattern` contains these fields, while `formula_pattern` does not.
            # Raises `IndexError` if `m` does not contain 'phase' or 'operation'.
            # Raises `KeyError` if `operation_to_role` does not contain the given combination of phase and operation.
            role = operation_to_role[m['phase'], m['operation']]
        except (IndexError, KeyError):
            role = default_role
        if role is None:
            continue
        if roles is not None and role not in roles:
            continue
        result = {
            'formula': {
                'id': int(m['formula_id']),
                'extra': extract_extra(m['extra'])
            },
            'role': role,
            'inference': {},
            'span': {
                'start': m.start(),
                'end': m.end()
            }
        }
        with suppress(IndexError):
            result['formula']['string'] = m['formula']
        with suppress(IndexError):
            result['inference']['rule'] = m['inference_rule']
        with suppress((IndexError, AttributeError)):
            # `m['inference_parents'].split()` raises AttributeError if `m['inference_parents']` is None.
            result['inference']['parents'] = list(map(int, m['inference_parents'].split(',')))
        yield result


def extract_extra(extra_str):
    if extra_str is None:
        return {}
    pair_strings = (p.strip() for p in extra_str.split(','))
    string_pairs = (p.split(':', 1) for p in pair_strings)
    return {k: cast_extra_value(k, v) for k, v in string_pairs}


def cast_extra_value(k, v):
    if k == 'wCS':
        return float(v)
    with suppress(ValueError):
        return int(v)
    with suppress(ValueError):
        return float(v)
    return v
