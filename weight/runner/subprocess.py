import logging
import subprocess
import time

import contexttimer


log = logging.getLogger(__name__)


def run(args, capture_stdout=True, capture_stderr=True, input='', **kwargs):
    log.debug('Running process: %s', ' '.join(args))
    with timer() as t:
        try:
            cp = subprocess.run(args,
                                input=input,
                                stdout=(subprocess.PIPE if capture_stdout else subprocess.DEVNULL),
                                stderr=(subprocess.PIPE if capture_stderr else subprocess.DEVNULL),
                                text=True,
                                **kwargs)
            terminationreason = None
            returncode = cp.returncode
            stdout = cp.stdout
            stderr = cp.stderr
        except subprocess.TimeoutExpired as e:
            terminationreason = 'walltime'
            returncode = None
            stdout = bytes_to_str(e.output)
            stderr = bytes_to_str(e.stderr)
    return {
        'args': args,
        'backend_kwargs': kwargs,
        'terminationreason': terminationreason,
        'returncode': returncode,
        'elapsed': t.elapsed,
        'stdout': stdout,
        'stderr': stderr
    }


def bytes_to_str(b):
    if b is None:
        return None
    if isinstance(b, bytes):
        return b.decode('utf-8')
    return str(b)


def timer(*args, **kwargs):
    return contexttimer.Timer(*args, timer=time.perf_counter, **kwargs)
