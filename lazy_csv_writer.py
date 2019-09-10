#!/usr/bin/env python3.7

import csv
import threading


class LazyCsvWriter:
    """
    A thread-safe csv.DictWriter that writes the header row automatically along with the first data row.
    """
    def __init__(self, file):
        self._file = file
        self._lock = threading.Lock()
        self._writer = None

    def writerow(self, row):
        """Write a CSV row and flush.

        All the calls to this instance must use the same field names.

        Mimics `csv.csvwriter.writerow`.

        :param row: a dictionary that maps field names to field values
        """
        with self._lock:
            if self._writer is None:
                self._writer = csv.DictWriter(self._file, fieldnames=row.keys())
                self._writer.writeheader()
            self._writer.writerow(row)
            self._file.flush()
