import logging
import subprocess
import time

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


def decode(b):
    if b is None:
        return None
    if isinstance(b, bytes):
        return b.decode('utf-8')
    return str(b)


def run(args, timeout=None, capture_stdout=True, capture_stderr=True):
    log.debug('Running process: %s', ' '.join(args))
    time_start = time.time()
    try:
        cp = subprocess.run(args,
                            stdout=(subprocess.PIPE if capture_stdout else subprocess.DEVNULL),
                            stderr=(subprocess.PIPE if capture_stderr else subprocess.DEVNULL), timeout=timeout,
                            universal_newlines=True)
        time_elapsed = time.time() - time_start
        returncode = cp.returncode
        stdout = cp.stdout
        stderr = cp.stderr
    except subprocess.TimeoutExpired as e:
        time_elapsed = time.time() - time_start
        returncode = None
        stdout = decode(e.output)
        stderr = decode(e.stderr)
    result = Result(returncode, time_elapsed, stdout, stderr)
    log.debug(result)
    return result
