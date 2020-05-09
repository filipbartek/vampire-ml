#!/usr/bin/env python3

import glob
import itertools
import logging
import os
from functools import partial

from ordered_set import OrderedSet


def normalize_path(path, base_path=None):
    if base_path is not None:
        path = os.path.join(base_path, path)
    return os.path.abspath(path)


def paths_from_sublists(sublist_file_paths, base_path=None):
    files = map(open, sublist_file_paths)
    line_generators = (file.readlines() for file in files)
    lines = itertools.chain(*line_generators)
    lines_stripped = (l.strip('\n') for l in lines)
    paths = map(partial(normalize_path, base_path=base_path), lines_stripped)
    return paths


def paths_from_patterns(patterns, base_path=None):
    # We modify the glob patterns to be relative to base_path.
    # Alternatively, we could change the CWD which is used by glob.iglob.
    patterns_normalized = map(partial(normalize_path, base_path=base_path), patterns)
    path_generators = map(partial(glob.iglob, recursive=True), patterns_normalized)
    paths = itertools.chain(*path_generators)
    return paths


def compose(sublist_file_paths=None, glob_patterns=None, base_path=None):
    paths = OrderedSet(itertools.chain(paths_from_sublists(sublist_file_paths, base_path=base_path),
                                       paths_from_patterns(glob_patterns, base_path=base_path)))
    if base_path is None and len(paths) >= 1:
        base_path = os.path.commonpath(paths)
        logging.info(f'Defaulting base path to \"{base_path}\".')
    return paths, base_path
