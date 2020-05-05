#!/usr/bin/env python3

import json
import logging
import os
import subprocess
import time


class Result:
    def __init__(self, time_elapsed, status, stdout, stderr, exit_code=None):
        self.time_elapsed = time_elapsed
        self.status = status
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code

    def __str__(self):
        return str({'status': self.status, 'exit_code': self.exit_code, 'time_elapsed': self.time_elapsed})

    def save(self, path):
        json.dump({'time_elapsed': self.time_elapsed, 'status': self.status, 'exit_code': self.exit_code},
                  open(os.path.join(path, 'result.json'), 'w'), indent=4)
        if self.stdout is not None:
            open(os.path.join(path, 'stdout.txt'), 'w').write(self.stdout)
        if self.stderr is not None:
            open(os.path.join(path, 'stderr.txt'), 'w').write(self.stderr)

    @staticmethod
    def load(path):
        data = json.load(open(os.path.join(path, 'result.json')))
        stdout = open(os.path.join(path, 'stdout.txt')).read()
        stderr = open(os.path.join(path, 'stderr.txt')).read()
        return Result(data['time_elapsed'], data['status'], stdout, stderr, data['exit_code'])


def run(args, timeout=None):
    time_start = time.time()
    try:
        logging.debug(' '.join(args))
        cp = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True)
        time_elapsed = time.time() - time_start
        status = 'completed'
        stdout = cp.stdout
        stderr = cp.stderr
        exit_code = cp.returncode
    except subprocess.TimeoutExpired as e:
        time_elapsed = time.time() - time_start
        status = 'timeout_expired'
        exit_code = None

        def decode(b):
            if b is None:
                return ''
            if isinstance(b, bytes):
                return b.decode('utf-8')
            return str(b)

        stdout = decode(e.output)
        stderr = decode(e.stderr)
    return Result(time_elapsed, status, stdout, stderr, exit_code=exit_code)
