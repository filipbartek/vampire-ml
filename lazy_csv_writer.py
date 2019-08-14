#!/usr/bin/env python3.7

import csv
import threading


class LazyCsvWriter:
    """
    A thread-safe csv.DictWriter that writes the header row automatically along with the first data row.
    """
    def __init__(self, file):
        self._lock = threading.Lock()
        self._csv_file = file
        self._writer = None

    def writerow(self, row):
        with self._lock:
            self.__ensure_writer(row.keys())
            self._writer.writerow(row)

    def __ensure_writer(self, fieldnames):
        if self._writer is None:
            self._writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._writer.writeheader()
