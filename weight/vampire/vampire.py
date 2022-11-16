import itertools
import json
import logging
import os
import re
from contextlib import suppress

from . import szs
from utils import to_str
from weight import runner

log = logging.getLogger(__name__)


def run(problem_path, options, out_dir=None, **kwargs):
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
        status_short = szs.from_output(output)
    result['szs_status'] = status_short

    with suppress(TypeError):
        result['elapsed_vampire'] = float(re.search(r'^% Time elapsed: (\d+\.\d+) s$', output, re.MULTILINE)[1])
    with suppress(TypeError):
        result['megainstructions'] = int(re.search(r'^% Instructions burned: (\d+) \(million\)$', output, re.MULTILINE)[1])
    with suppress(TypeError):
        result['activations'] = int(re.search(r'^% Activations started: (\d+)$', output, re.MULTILINE)[1])
    with suppress(TypeError):
        result['passive'] = int(re.search(r'^% Passive clauses: (\d+)$', output, re.MULTILINE)[1])
    with suppress(TypeError):
        result['memory'] = int(re.search(r'^% Memory used \[KB\]: (\d+)$', output, re.MULTILINE)[1])

    # 523837 Aborted by signal SIGHUP on /home/bartefil/TPTP-v7.5.0/Problems/SYN/SYN764-1.p
    # Aborted by signal SIGTERM on /home/filip/TPTP-v7.5.0/Problems/SET/SET713+4.p
    m = re.search(r'^(?:(?P<pid>\d+) )?Aborted by signal (?P<signal>\w+) on (?P<problem>.+)$', output, re.MULTILINE)
    if m is not None:
        result['signal'] = m['signal']
        result['pid'] = m['pid']
    result['cmd'] = ' '.join(result['args'])
    save_result(out_dir, result)
    result['out_dir'] = out_dir
    m = re.search('^User error: Cannot open problem file: (?P<problem>.*)$', output, re.MULTILINE)
    if m is not None:
        raise RuntimeError(f'Cannot open problem file: {problem_path}')
    return result


def run_bare(problem, options, vampire='vampire', **kwargs):
    args = [vampire, problem, *itertools.chain.from_iterable((f'--{k}', to_str(v)) for k, v in options.items())]
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


def save_result(out_dir, result):
    if out_dir is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump({k: v for k, v in result.items() if k not in ['stdout', 'stderr']}, f, indent=4, default=str)
        log.debug(f'{f.name} saved.')
    if 'stdout' in result:
        with open(os.path.join(out_dir, 'stdout.txt'), 'w') as f:
            f.write(result['stdout'])
            log.debug(f'{f.name} saved.')
    if 'stderr' in result:
        with open(os.path.join(out_dir, 'stderr.txt'), 'w') as f:
            f.write(result['stderr'])
            log.debug(f'{f.name} saved.')
