import itertools
import json
import os
import re
from contextlib import suppress

from weight import runner


def run(problem_path, options, out_dir, **kwargs):
    result = run_bare(problem_path, options, **kwargs)
    status_short = None
    with suppress(KeyError):
        terminationreason = result['terminationreason']
        if terminationreason == 'failed':
            raise RuntimeError('Failed to execute Vampire.')
        if terminationreason in ['walltime', 'cputime', 'cputime-soft']:
            status_short = 'TMO'
    output = result['stdout']
    if status_short is None:
        status_short = extract_status(output)
    result['szs_status'] = status_short
    save_result(out_dir, result)
    result['out_dir'] = out_dir
    m = re.search('^User error: Cannot open problem file: (?P<problem>.*)$', output, re.MULTILINE)
    if m is not None:
        raise RuntimeError(f'Cannot open problem file: {problem_path}')
    return result


def run_bare(problem, options, vampire='vampire', **kwargs):
    args = [vampire, problem, *itertools.chain.from_iterable((f'--{k}', str(v)) for k, v in options.items())]
    result = runner.run(args, **kwargs)
    result['problem'] = problem
    result['vampire'] = {
        'vampire': vampire,
        'options': options
    }
    if 'stdout' in result:
        result['stdout_len'] = len(result['stdout'])
    if 'stderr' in result:
        result['stderr_len'] = len(result['stderr'])
    return result


def extract_status(output):
    status_long = szs_status(output)
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
    'IAP': 'Inappropriate'
}

long_to_short = {v: k for k, v in short_to_long.items()}


def szs_status(stdout):
    # http://www.tptp.org/TPTP/TPTPTParty/2007/PositionStatements/GeoffSutcliffe_SZS.html
    # Example:
    # % SZS status Unsatisfiable for SET846-1
    # `vampire --mode casc` always outputs exactly one SZS status.
    m = re.search(r'^% SZS status (?P<status>.*) for (?P<problem>.*)$', stdout, re.MULTILINE)
    if m is not None:
        return m['status']
    return None


def save_result(out_dir, result):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump({k: v for k, v in result.items() if k not in ['stdout', 'stderr']}, f, indent=4, default=str)
    if 'stdout' in result:
        with open(os.path.join(out_dir, 'stdout.txt'), 'w') as f:
            f.write(result['stdout'])
    if 'stderr' in result:
        with open(os.path.join(out_dir, 'stderr.txt'), 'w') as f:
            f.write(result['stderr'])


def termination_reason(stdout):
    m = re.search(r'^% Termination reason: (.*)$', stdout, re.MULTILINE)
    if m is not None:
        return m[1]
    return None


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
