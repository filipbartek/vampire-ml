import logging
import re
from contextlib import suppress

from . import szs

log = logging.getLogger(__name__)


def from_output(stdout, *args, **kwargs):
    result = szs.from_output(stdout, *args, **kwargs)

    with suppress(TypeError):
        result['elapsed'] = float(re.search(r'^% Time elapsed: (\d+\.\d+) s$', stdout, re.MULTILINE)[1])
    with suppress(TypeError):
        result['megainstructions'] = int(
            re.search(r'^% Instructions burned: (\d+) \(million\)$', stdout, re.MULTILINE)[1])
    try:
        result['activations'] = int(re.search(r'^% Activations started: (\d+)$', stdout, re.MULTILINE)[1])
    except TypeError:
        log.debug('No activations.')
        result['activations'] = 0
    try:
        result['passive'] = int(re.search(r'^% Passive clauses: (\d+)$', stdout, re.MULTILINE)[1])
    except TypeError:
        result['passive'] = 0
    with suppress(TypeError):
        result['memory'] = int(re.search(r'^% Memory used \[KB\]: (\d+)$', stdout, re.MULTILINE)[1])
    result['termination'] = {}
    with suppress(TypeError):
        result['termination']['reason'] = re.search(r'^% Termination reason: (\w+)$', stdout, re.MULTILINE)[1]
    with suppress(TypeError):
        result['termination']['phase'] = re.search(r'^% Termination phase: ([\w ]+)$', stdout, re.MULTILINE)[1]

    # 523837 Aborted by signal SIGHUP on /home/bartefil/TPTP-v7.5.0/Problems/SYN/SYN764-1.p
    # Aborted by signal SIGTERM on /home/filip/TPTP-v7.5.0/Problems/SET/SET713+4.p
    m = re.search(r'^(?:(?P<pid>\d+) )?Aborted by signal (?P<signal>\w+) on (?P<problem>.+)$', stdout, re.MULTILINE)
    if m is not None:
        result['signal'] = m['signal']
        result['pid'] = m['pid']

    return result
