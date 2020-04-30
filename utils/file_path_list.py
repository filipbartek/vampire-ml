#!/usr/bin/env python3

import glob
import itertools
import logging
import os


def compose(sublist_file_paths=None, glob_patterns=None, base_path=None):
    file_paths = []
    if sublist_file_paths is not None:
        for sublist_file_path in sublist_file_paths:
            with open(sublist_file_path) as sublist_file:
                for l in sublist_file.readlines():
                    cur_path = l.rstrip('\n')
                    if base_path is not None:
                        cur_path = os.path.join(base_path, cur_path)
                    cur_path = os.path.abspath(cur_path)
                    file_paths.append(cur_path)
    # We modify the glob patterns to be relative to base_path.
    # Alternatively, we could change the CWD which is used by glob.iglob.
    if glob_patterns is not None:
        glob_patterns_rel_to_base = glob_patterns
        if base_path is not None:
            glob_patterns_rel_to_base = (os.path.join(base_path, pattern) for pattern in glob_patterns_rel_to_base)
        file_paths.extend(
            itertools.chain(*(glob.iglob(pattern, recursive=True) for pattern in glob_patterns_rel_to_base)))
    if base_path is None and len(file_paths) >= 1:
        base_path = os.path.commonpath(file_paths)
        logging.info(f'Defaulting base path to \"{base_path}\".')
    return file_paths, base_path
