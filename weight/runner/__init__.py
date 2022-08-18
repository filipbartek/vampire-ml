from . import benchexec
from . import subprocess


def run(args, backend='subprocess', **kwargs):
    if backend == 'subprocess':
        result = subprocess.run(args, **kwargs)
    elif backend == 'benchexec':
        result = benchexec.run(args, **kwargs)
    else:
        raise ValueError(f'Unknown backend: {backend}')
    result['backend'] = backend
    return result
