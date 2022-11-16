# https://www.tptp.org/TPTP/TPTPTParty/2007/PositionStatements/GeoffSutcliffe_SZS.html

import re

short_to_long = {
    # Unsatisfiable
    'THM': 'Theorem',
    'CAX': 'ContradictoryAxioms',
    'UNS': 'Unsatisfiable',
    # Satisfiable
    'SAT': 'Satisfiable',
    'CSA': 'CounterSatisfiable',
    # Unsolved
    'UNK': 'Unknown',
    'ERR': 'Error',
    'TMO': 'Timeout',
    'MMO': 'MemoryOut',
    'GUP': 'GaveUp',
    'INC': 'Incomplete',
    'IAP': 'Inappropriate',
    # Non-standard statuses
    'ACO': 'ActivationsOut',
    'INO': 'InstructionsOut'
}

long_to_short = {v: k for k, v in short_to_long.items()}

termination_reason_to_status = {
    'Refutation': 'UNS',
    'Satisfiable': 'SAT',
    'Time limit': 'TMO',
    'Memory limit': 'MMO',
    'Activation limit': 'ACO',
    'Refutation not found, non-redundant clauses discarded': 'INC',
    'Refutation not found, incomplete strategy': 'INC',
    'Inappropriate': 'IAP',
    'Unknown': 'UNK'
}


def from_output(output):
    status_long = long_from_output_field(output)
    try:
        return long_to_short[status_long]
    except KeyError:
        term_reason = termination_reason(output)
        if term_reason == 'Unknown':
            if re.search(r'% Instruction limit reached!$', output, re.MULTILINE):
                return 'INO'
        try:
            return termination_reason_to_status[term_reason]
        except KeyError:
            return 'UNK'


def long_from_output_field(stdout):
    # http://www.tptp.org/TPTP/TPTPTParty/2007/PositionStatements/GeoffSutcliffe_SZS.html
    # Example:
    # % SZS status Unsatisfiable for SET846-1
    # `vampire --mode casc` always outputs exactly one SZS status.
    m = re.search(r'^% SZS status (?P<status>.*) for (?P<problem>.*)$', stdout, re.MULTILINE)
    if m is not None:
        return m['status']
    return None


def termination_reason(stdout):
    m = re.search(r'^% Termination reason: (.*)$', stdout, re.MULTILINE)
    if m is not None:
        return m[1]
    return None
