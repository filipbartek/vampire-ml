import logging

from . import benchexec
from . import subprocess

log = logging.getLogger(__name__)


def run(args, backend='subprocess', **kwargs):
    log.debug(f'Running with {backend}: %s' % ' '.join(args))
    if backend == 'subprocess':
        result = subprocess.run(args, **kwargs)
    elif backend == 'benchexec':
        result = benchexec.run(args, **kwargs)
    else:
        raise ValueError(f'Unknown backend: {backend}')
    result['backend'] = backend
    return result
