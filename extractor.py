#!/usr/bin/env python3.7

import logging
import re


class ReExtractor:
    def __init__(self, pattern, convert, name):
        """
        :param pattern: A regular expression object.
        :param convert: A function that converts a regular expression match object to the value of this match.
        """
        self.pattern = pattern
        self.convert = convert
        self.name = name

    def __call__(self, s):
        m = self.pattern.search(s)
        if m is None:
            raise RuntimeError(f'Pattern {self.name} failed to match.')
        return self.convert(m)


class MultiExtractor:
    def __init__(self, extractors):
        """
        :param extractors: a dictionary indexed by extractor names. Each value is a pair of an extractor and a boolean flag that is True iff that partial extractor is required.
        """
        self.extractors = extractors

    def __call__(self, s):
        result = dict()
        for name, (extractor, required) in self.extractors.items():
            try:
                result[name] = extractor(s)
            except RuntimeError as err:
                if required:
                    logging.debug(f'Extracting {name} failed: {err}')
                result[name] = None
        return result


vampire_version = ReExtractor(re.compile(
    '^% Version: Vampire (?P<number>[\d\.]*) \(commit (?P<commit_hash>[\da-f]*) on (?P<commit_timestamp>.*)\)$',
    re.MULTILINE), lambda m: m.groupdict(), 'vampire_version')

termination_reason = ReExtractor(re.compile('^% Termination reason: (.*)$', re.MULTILINE), lambda m: m[1],
                                 'termination_reason')
termination_phase = ReExtractor(re.compile('^% Termination phase: (.*)$', re.MULTILINE), lambda m: m[1],
                                'termination_phase')
termination = MultiExtractor({
    'reason': (termination_reason, True),
    'phase': (termination_phase, False)
})

function_names = ReExtractor(re.compile('^% Function symbol names: ([\w,]*)$', re.MULTILINE),
                             lambda m: list(m[1].split(',')), 'function_names')
function_precedence = ReExtractor(
    re.compile('^% Function symbol index precedence permutation: ([\d,]*)$', re.MULTILINE),
    lambda m: list(map(int, m[1].split(','))), 'function_precedence')
predicate_names = ReExtractor(re.compile('^% Predicate symbol names: ([\w=,]*)$', re.MULTILINE),
                              lambda m: list(m[1].split(',')), 'predicate_names')
predicate_precedence = ReExtractor(
    re.compile('^% Predicate symbol index precedence permutation: ([\d,]*)$', re.MULTILINE),
    lambda m: list(map(int, m[1].split(','))), 'predicate_precedence')

saturation_initial_clauses = ReExtractor(re.compile('^% Initial clauses: (\d+)$', re.MULTILINE),
                                         lambda m: int(m[1]), 'initial_clauses')
saturation_generated_clauses = ReExtractor(re.compile('^% Generated clauses: (\d+)$', re.MULTILINE),
                                           lambda m: int(m[1]), 'generated_clauses')
saturation_active_clauses = ReExtractor(re.compile('^% Active clauses: (\d+)$', re.MULTILINE),
                                        lambda m: int(m[1]), 'active_clauses')
saturation_passive_clauses = ReExtractor(re.compile('^% Passive clauses: (\d+)$', re.MULTILINE),
                                         lambda m: int(m[1]), 'passive_clauses')
saturation_final_active_clauses = ReExtractor(re.compile('^% Final active clauses: (\d+)$', re.MULTILINE),
                                              lambda m: int(m[1]), 'final_active_clauses')
saturation_final_passive_clauses = ReExtractor(re.compile('^% Final passive clauses: (\d+)$', re.MULTILINE),
                                               lambda m: int(m[1]), 'final_passive_clauses')
saturation_iterations = ReExtractor(re.compile('^% Main loop iterations: (\d+)$', re.MULTILINE),
                                    lambda m: int(m[1]), 'saturation_iterations')
saturation = MultiExtractor({
    'initial_clauses': (saturation_initial_clauses, True),
    'generated_clauses': (saturation_generated_clauses, True),
    'active_clauses': (saturation_active_clauses, True),
    'passive_clauses': (saturation_passive_clauses, True),
    'final_active_clauses': (saturation_final_active_clauses, True),
    'final_passive_clauses': (saturation_final_passive_clauses, True),
    'iterations': (saturation_iterations, True)
})

memory_used = ReExtractor(re.compile('^% Memory used \[KB\]: (\d+)$', re.MULTILINE), lambda m: int(m[1]), 'memory_used')
time_elapsed = ReExtractor(re.compile('^% Time elapsed: (\d+\.\d+) s$', re.MULTILINE), lambda m: float(m[1]),
                           'time_elapsed')

complete = MultiExtractor({
    'vampire_version': (vampire_version, True),
    'strategy': (lambda s: s.partition('\n')[0], True),
    'termination': (termination, True),
    'time_elapsed': (time_elapsed, True),
    'memory_used': (memory_used, True),
    'predicate_names': (predicate_names, True),
    'predicate_precedence': (predicate_precedence, True),
    'function_names': (function_names, True),
    'function_precedence': (function_precedence, True),
    'saturation': (saturation, True)
})

clausify = MultiExtractor({
    'strategy': (lambda s: s.partition('\n')[0], True),
    'vampire_version': (vampire_version, False),
    'termination': (termination, False),
    'time_elapsed': (time_elapsed, False)
})

# TODO: Parse run statistics, namely activation count.
