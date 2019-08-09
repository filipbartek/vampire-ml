#!/usr/bin/env python3.7

import json
import logging
import os
import subprocess
import time

import extractor


class Vampire:
    def __init__(self, command, file_names=None):
        self._command = command
        if file_names is None:
            file_names = {
                'stdout': 'stdout.txt',
                'stderr': 'stderr.txt',
                'data': 'data.json'
            }
        self._file_names = file_names

    def __call__(self, parameters, problem_path, output_path):
        """
        Run Vampire once.
        :param parameters: parameters recognized by compose_args()
        :return: result dictionary
        """
        vampire_args = self.compose_args(parameters, problem_path)
        complete_command = ' '.join(vampire_args)
        logging.info(complete_command)
        time_start = time.time()
        cp = subprocess.run(vampire_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time_elapsed = time.time() - time_start
        run_data = self.get_run_data(parameters, output_path, problem_path, complete_command, cp, time_elapsed)
        data_path = self.save_to_file(output_path, 'data', lambda outfile: json.dump(run_data, outfile, indent=4))
        return run_data, data_path

    def get_run_data(self, parameters, output_path, problem_path, complete_command, cp, time_elapsed):
        extract = extractor.complete
        if 'mode' in parameters and parameters['mode'] == 'clausify':
            extract = extractor.clausify
        vampire_json_output_path = None
        if 'json_output' in parameters:
            vampire_json_output_path = os.path.relpath(parameters['json_output'], start=output_path)
        return {
            'vampire': self._command,
            'vampire_parameters': parameters,
            'problem_path': problem_path,
            'output_path': output_path,
            'call': {
                'cwd': os.getcwd(),
                'command': complete_command,
                'exit_code': cp.returncode,
                'time_elapsed': time_elapsed
            },
            'output': {
                'stdout': self.save_to_file(output_path, 'stdout', lambda outfile: outfile.write(cp.stdout)),
                'stderr': self.save_to_file(output_path, 'stderr', lambda outfile: outfile.write(cp.stderr)),
                'json_output_path': vampire_json_output_path,
                'data': extract(cp.stdout)
            }
        }

    def compose_args(self, parameters, problem_path):
        """
        See vampire_options.
        :return: list of arguments compatible with subprocess.run().
        """
        args = [self._command]
        for key, value in parameters.items():
            if key not in self.vampire_options:
                logging.warning(f'Unknown Vampire option: {key}')
                continue
            if value is None:
                continue
            args.extend([self.vampire_options[key], str(value)])
        args.append(problem_path)
        return args

    vampire_options = {
        'include': '--include',
        'time_limit': '--time_limit',
        'proof': '--proof',
        'symbol_precedence': '--symbol_precedence',
        'encode': '--encode',
        'statistics': '--statistics',
        'time_statistics': '--time_statistics',
        'random_seed': '--random_seed',
        'json_output': '--json_output',
        'mode': '--mode'
    }

    def save_to_file(self, output_path, file_id, save_fn):
        try:
            file_path = os.path.join(output_path, self._file_names[file_id])
        except KeyError:
            # Fail gracefully if this file id is not supported.
            return None
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as outfile:
            save_fn(outfile)
        return os.path.relpath(file_path, start=output_path)
