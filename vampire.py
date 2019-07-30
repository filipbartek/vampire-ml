#!/usr/bin/env python3.7

import json
import logging
import os
import subprocess

import extractor


class Vampire:
    def __init__(self, command):
        self.command = command

    def __call__(self, parameters):
        """
        Run Vampire once.
        :param parameters: Dictionary of execution parameters. Structure of parameters:
        * vampire: parameters recognized by compose_args()
        * paths
            * problem: problem input file path (TPTP)
            * stdout: stdout output file path (text)
            * stderr: stderr output file path (text)
            * data: data output file path (JSON)
        :return: result dictionary.
        """
        vampire_args = self.compose_args(parameters['vampire'], parameters['paths']['problem'])
        complete_command = ' '.join(vampire_args)
        logging.info(complete_command)
        cp = subprocess.run(vampire_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        run_data = {
            'parameters': parameters,
            'call': {
                'cwd': os.getcwd(),
                'command': complete_command
            },
            'output': {
                'exit_code': cp.returncode,
                'data': extractor.complete(cp.stdout)
            }
        }
        # TODO: Make the paths in the result JSON document relative to the document.
        self.save_str(parameters['paths']['stdout'], cp.stdout)
        self.save_str(parameters['paths']['stderr'], cp.stderr)
        if parameters['paths']['data'] is not None:
            os.makedirs(os.path.dirname(parameters['paths']['data']), exist_ok=True)
            with open(parameters['paths']['data'], 'w') as outfile:
                json.dump(run_data, outfile, indent=4)
        return run_data

    def compose_args(self, parameters, problem_path):
        """
        See vampire_options.
        :return: list of arguments compatible with subprocess.run().
        """
        args = [self.command]
        for key, value in parameters.items():
            if key in self.vampire_options:
                args.extend([self.vampire_options[key], str(value)])
            else:
                logging.warning(f'Unknown Vampire option: {key}')
        args.append(problem_path)
        return args

    vampire_options = {
        'include': '--include',
        'time_limit': '--time_limit',
        'proof': '--proof',
        'symbol_precedence': '--symbol_precedence',
        'encode': '--encode',
        'statistics': '--statistics',
        'random_seed': '--random_seed',
        'json_output': '--json_output',
        'mode': '--mode'
    }

    @staticmethod
    def save_str(outfilepath, s):
        if outfilepath is not None:
            os.makedirs(os.path.dirname(outfilepath), exist_ok=True)
            with open(outfilepath, 'w') as outfile:
                outfile.write(s)
        return outfilepath
