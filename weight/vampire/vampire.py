import itertools
import json
import logging
import os
import re

from . import stats
from utils import to_str
from weight import runner

log = logging.getLogger(__name__)


def run(problem_path, options, out_dir=None, **kwargs):
    result = run_bare(problem_path, options, **kwargs)
    result['cmd'] = ' '.join(result['args'])
    terminationreason = result.get('terminationreason')
    if terminationreason == 'failed':
        result['szs_status'] = 'ERR'
        result['error'] = 'Failed to execute Vampire.'
    result.update({{'elapsed': 'vampire_elapsed'}.get(k, k): v for k, v in stats.from_output(result['stdout']).items()})
    if result['szs_status'] == 'INE':
        result['error'] = 'Problem file not found.'
    m = re.search('^perf_event_open failed \(instruction limiting will be disabled\): Permission denied$', result['stderr'], re.MULTILINE)
    if terminationreason in ['walltime', 'cputime', 'cputime-soft']:
        assert result['szs_status'] in ['TMO', 'UNK']
        result['szs_status'] = 'TMO'
    if m is not None:
        result['szs_status'] = 'OSE'
        result['error'] = 'Instruction limiting is disabled.'
    save_result(out_dir, result)
    result['out_dir'] = out_dir
    return result


def run_bare(problem, options, vampire='vampire', **kwargs):
    args = [vampire, problem, *itertools.chain.from_iterable((f'--{k}', to_str(v)) for k, v in options.items())]
    result = runner.run(args, **kwargs)
    result['problem'] = problem
    result['vampire'] = {
        'vampire': vampire,
        'options': options
    }
    if 'stdout' in result and result['stdout'] is not None:
        result['stdout_len'] = len(result['stdout'])
    if 'stderr' in result and result['stderr'] is not None:
        result['stderr_len'] = len(result['stderr'])
    return result


def save_result(out_dir, result):
    if out_dir is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump({k: v for k, v in result.items() if k not in ['stdout', 'stderr']}, f, indent=4, default=str)
        log.debug(f'{f.name} saved.')
    if 'stdout' in result and result['stdout'] is not None:
        with open(os.path.join(out_dir, 'stdout.txt'), 'w') as f:
            f.write(result['stdout'])
            log.debug(f'{f.name} saved.')
    if 'stderr' in result and result['stderr'] is not None:
        with open(os.path.join(out_dir, 'stderr.txt'), 'w') as f:
            f.write(result['stderr'])
            log.debug(f'{f.name} saved.')
