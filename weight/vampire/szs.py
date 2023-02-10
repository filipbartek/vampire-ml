# https://www.tptp.org/TPTP/TPTPTParty/2007/PositionStatements/GeoffSutcliffe_SZS.html

import re
from contextlib import suppress

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
    'INE': 'InputError',
    'OSE': 'OSError',
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


def from_output(stdout, stderr, terminationreason=None):
    if terminationreason == 'failed':
        return {'szs_status': 'ERR', 'error': {'category': 'Failed to execute Vampire'}}

    m = re.search('^perf_event_open failed \(instruction limiting will be disabled\): Permission denied$', stderr,
                  re.MULTILINE)
    if m is not None:
        return {'szs_status': 'OSE', 'error': {'category': 'Instruction limiting is disabled', 'message': m[0]}}
    m = re.search('^User error: Cannot open problem file: (?P<problem>.*)$', stdout, re.MULTILINE)
    if m is not None:
        return {'szs_status': 'INE', 'error': {'category': 'Cannot open problem file', 'message': m[0]}}

    m = re.search('^Parsing Error on line (?P<line>\d+)\n.*$', stdout, re.MULTILINE)
    if m is not None:
        return {'szs_status': 'INE', 'error': {'category': 'Parsing error', 'message': m[0]}}
    
    if terminationreason in ['walltime', 'cputime', 'cputime-soft']:
        return {'szs_status': 'TMO'}

    status_long = long_from_output_field(stdout)
    with suppress(KeyError):
        return {'szs_status': long_to_short[status_long]}
    term_reason = termination_reason(stdout)
    if term_reason == 'Unknown':
        if re.search(r'% Instruction limit reached!$', stdout, re.MULTILINE):
            return {'szs_status': 'INO'}
    with suppress(KeyError):
        return {'szs_status': termination_reason_to_status[term_reason]}
    return {'szs_status': 'UNK', 'error': {'category': 'Unable to determine the status'}}


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


def is_unsat(status):
    return status in ['THM', 'CAX', 'UNS']


def is_sat(status):
    return status in ['SAT', 'CSA']


def is_solved(status):
    return is_unsat(status) or is_sat(status)
