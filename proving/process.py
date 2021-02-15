import logging
import subprocess

from proving import utils

log = logging.getLogger(__name__)


class Result:
    def __init__(self, returncode, time_elapsed, stdout, stderr):
        self.returncode = returncode
        self.time_elapsed = time_elapsed
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self):
        return utils.instance_str(self)

    pd_dtypes = {
        'returncode': 'category',
        'time_elapsed': float
    }

    def as_record(self):
        return {k: getattr(self, k) for k in self.pd_dtypes}


def decode(b):
    if b is None:
        return None
    if isinstance(b, bytes):
        return b.decode('utf-8')
    return str(b)


def run(args, timeout=None, capture_stdout=True, capture_stderr=True):
    log.debug('Running process: %s', ' '.join(args))
    with utils.timer() as t:
        try:
            cp = subprocess.run(args,
                                stdout=(subprocess.PIPE if capture_stdout else subprocess.DEVNULL),
                                stderr=(subprocess.PIPE if capture_stderr else subprocess.DEVNULL), timeout=timeout,
                                universal_newlines=True)
            returncode = cp.returncode
            stdout = cp.stdout
            stderr = cp.stderr
        except subprocess.TimeoutExpired as e:
            returncode = None
            stdout = decode(e.output)
            stderr = decode(e.stderr)
    result = Result(returncode, t.elapsed, stdout, stderr)
    log.debug(result)
    return result
