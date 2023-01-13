import re
from collections import Counter


def parse(clause):
    m = prog.fullmatch(clause)

    return {polarity: {
        'predicate': str_to_counter(m[f'{polarity}_predicate']),
        'function': str_to_counter(m[f'{polarity}_function']),
        'variable': int(m[f'{polarity}_variable'])
    } for polarity in ['positive', 'negative']}


def get_pattern_count_map(name):
    return fr'\{{(?P<{name}>[\d:, ]*)\}}'


def get_pattern_literal(name):
    pattern_predicate = get_pattern_count_map(f'{name}_predicate')
    pattern_function = get_pattern_count_map(f'{name}_function')
    pattern_variable = fr'(?P<{name}_variable>\d+)'
    return fr'\{{predicate: {pattern_predicate}, variable: {pattern_variable}, function: {pattern_function}\}}'


def get_pattern():
    pattern_positive = get_pattern_literal('positive')
    pattern_negative = get_pattern_literal('negative')
    return fr'\{{positive: {pattern_positive}, negative: {pattern_negative}\}}'


prog = re.compile(fr'^{get_pattern()}$')


def str_to_counter(s):
    str_items = list(i.strip() for i in s.split(','))
    str_items_nonempty = list(i for i in str_items if len(i) > 0)
    str_pairs = list(p.split(':', 1) for p in str_items_nonempty)
    str_pairs_valid = list(p for p in str_pairs if len(p) == 2)
    return Counter({int(k.strip()): int(v.strip()) for k, v in str_pairs_valid})
