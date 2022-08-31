from collections import Counter

import pyparsing as pp


# https://www.tptp.org/TPTP/SyntaxBNF.html
alpha_numeric = pp.alphanums + '_'
upper_word = pp.Word(pp.alphas.upper(), alpha_numeric)
variable = upper_word.set_results_name('variable', list_all_matches=True)
lower_word = pp.Word(pp.alphas.lower(), alpha_numeric)
functor = lower_word.set_results_name('symbol', list_all_matches=True)
fof_term = pp.Forward()
fof_plain_term = functor + pp.Optional('(' + pp.delimited_list(fof_term, ',') + ')')
fof_term <<= variable | fof_plain_term
infix_equality = pp.Literal('=').set_results_name('equality', list_all_matches=True)
infix_inequality = pp.Literal('!=').set_results_name('inequality', list_all_matches=True)
infix_op = infix_equality | infix_inequality
infix_atom = fof_term + infix_op + fof_term
fof_atomic_formula = infix_atom | fof_plain_term
negation = pp.Literal('~').set_results_name('negation', list_all_matches=True)
literal = pp.Optional(negation) + fof_atomic_formula
cnf_formula = pp.delimited_list(literal, '|')


def token_counts(c):
    parsed = cnf_formula.parse_string(c, parse_all=True)
    res = {}
    if 'symbol' in parsed.keys():
        res['symbol'] = Counter(iter(parsed['symbol']))
    else:
        res['symbol'] = Counter()
    if 'variable' in parsed.keys():
        res['variable'] = sorted(Counter(iter(parsed['variable'])).values(), reverse=True)
    else:
        res['variable'] = []
    for name in ['equality', 'inequality', 'negation']:
        if name in parsed.keys():
            res[name] = len(parsed[name])
        else:
            res[name] = 0
    return res
