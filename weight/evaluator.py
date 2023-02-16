import json
import logging
import os
import sys
import tempfile
import warnings
from contextlib import contextmanager
from contextlib import suppress

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
import yaml

from questions.memory import memory
from questions.utils import timer
from utils import get_parallel
from utils import get_verbose
from utils import path_join
from utils import to_str
from vampire import szs
from weight import dataset
from weight import proof
from weight import vampire


def evaluate_one(evaluator, problem, weight, cur_out_dir):
    try:
        with timer() as t:
            result = {
                'problem': problem,
                'weight': weight,
                'out_dir': cur_out_dir,
                **evaluator.evaluate_one(problem, weight, out_dir=cur_out_dir)
            }
        result['time'] = t.elapsed
        return result
    except Exception as e:
        raise RuntimeError(f'Problem: {problem}') from e


class Evaluator:
    def evaluate(self, problem_weights, out_dir=None):
        problems, weights = zip(*problem_weights)
        if len(problems) > 1:
            base_path = os.path.commonpath(problems)
        else:
            base_path = os.path.dirname(problems[0])

        allunique = len(set(problems)) == len(problems)
        
        def get_out_dir(problem, i):
            if out_dir is None:
                return None
            result = os.path.join(out_dir, os.path.relpath(problem, base_path))
            if not allunique:
                result = os.path.join(result, str(i))
            return result
        
        cases = list(zip(problems, weights, (get_out_dir(problem, i) for i, problem in enumerate(problems))))

        if get_verbose():
            print(f'Evaluating {len(problem_weights)} weight vectors', file=sys.stderr)
        with get_parallel(len(problem_weights)) as parallel:
            with timer() as t:
                result = parallel(joblib.delayed(evaluate_one)(self, problem, weight, cur_out_dir) for problem, weight, cur_out_dir in cases)
            logging.debug(f'Time to evaluate {len(problem_weights)} weight vectors: {t.elapsed}')
            return result

    def evaluate_one(self, problem, weight, out_dir=None, iteration=None):
        raise NotImplemented

    @classmethod
    def get_problem_out_dir(cls, out_dir, problem):
        return path_join(out_dir, cls.get_problem_basename(problem))

    @staticmethod
    def get_problem_basename(problem):
        return problem.replace('/', '_')


class Composite(Evaluator):
    def __init__(self, evaluators, **kwargs):
        super().__init__(**kwargs)
        self.evaluators = evaluators

    def evaluate_one(self, problem, weight, out_dir=None, iteration=None):
        result = {name: evaluator.evaluate_one(problem, weight, out_dir=path_join(out_dir, name),
                                               iteration=path_join(name, iteration)) for name, evaluator
                  in self.evaluators.items()}
        if out_dir is not None:
            with open(path_join(out_dir, 'result.json')) as f:
                json.dump(result, f, indent=4)
        return result


class Proxy(Evaluator):
    def __init__(self, data):
        self.data = data

    def evaluate_one(self, problem, weight, out_dir=None, iteration=None):
        proofs = self.data[problem]
        ds = dataset.proofs_to_samples(proofs)
        return {
            'pair_accuracy': np.average((ds['pair']['X'] * weight >= 0) == ds['pair']['y'],
                                        weights=ds['pair']['sample_weight']),
            'single_roc_auc': sklearn.metrics.roc_auc_score(ds['single']['y'], ds['single']['X'] * weight,
                                                            sample_weight=ds['single']['sample_weight'])
        }


@memory.cache(ignore=['out_dir'])
def empirical_evaluate_one(evaluator, problem, weight, out_dir):
    return evaluator._evaluate_one(problem, weight, out_dir)


class Empirical(Evaluator):
    def __init__(self, runner_probe, clausifier, clause_features, runner_verbose=None, szs_status_of_interest=None,
                 plot_max_features=0):
        self.runner_probe = runner_probe
        self.runner_verbose = runner_verbose
        self.clausifier = clausifier
        self.clause_features = clause_features
        if szs_status_of_interest is None:
            szs_status_of_interest = ['THM', 'CAX', 'UNS'] + ['TMO', 'MMO', 'GUP', 'INC', 'ACO', 'INO']
        self.szs_status_of_interest = szs_status_of_interest
        self.plot_max_features = plot_max_features

    def evaluate_one(self, problem, weight, out_dir=None, iteration=None):
        result = {}
        if iteration is not None and len(weight) <= self.plot_max_features:
            with timer() as t:
                weights_dict, symbols = self.weight_vector_to_dict(problem, weight)
                # weights_dict['symbol']['='] = weights_dict['equality']
                symbols['weight'] = [weights_dict['equality']] + list(weights_dict['symbol'].values())[1:]
                features = symbols.copy()
                features.reset_index(inplace=True)
                features = features.astype({'id': pd.UInt64Dtype()}, copy=False)

                def symbol_to_category(symbol):
                    if symbol.name == '=':
                        return 'equality'
                    cat = {False: 'predicate', True: 'function'}[symbol.isFunction]
                    if symbol.introduced:
                        cat += '_introduced'
                    return cat

                cats = ['variable', 'equality', 'predicate', 'predicate_introduced', 'function', 'function_introduced']
                features['category'] = pd.Series([symbol_to_category(s) for s in features.itertuples(index=False)],
                                                 dtype=pd.CategoricalDtype(cats))

                var_record = {'name': '$var', 'weight': weights_dict['variable_occurrence'], 'isFunction': True,
                              'arity': 0, 'category': 'variable'}
                features = pd.concat([features, pd.DataFrame.from_records([var_record])])
                features['name_arity'] = ['%s_%u' % (f.name, f.arity) for f in features.itertuples(index=False)]

                features.sort_values('weight', inplace=True)
                plt.figure(figsize=(16, 12))
                sns.scatterplot(data=features, y='name_arity', x='weight', hue='category', palette='Paired')
                filename = os.path.join('weight', f'{iteration}.svg')
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)
                plt.close()
            result['plot_time'] = t.elapsed
        result.update(empirical_evaluate_one(self, problem, weight, out_dir))
        with suppress(KeyError):
            if result['probe']['szs_status'] == 'OSE':
                raise RuntimeError(result['probe']['error'])
        return result

    def _evaluate_one(self, problem, weight, out_dir):
        try:
            weights_dict, symbols = self.weight_vector_to_dict(problem, weight)
        except RuntimeError as e:
            warnings.warn(f'{problem}: {e}')
            return {'error': str(e)}
        signature = symbols.name
        result = {}
        with weight_options(weights_dict, path_join(out_dir, 'functor_weight.txt')) as options:
            result['probe'] = self.result_to_record(
                self.runner_probe.run(problem, options, out_dir=path_join(out_dir, 'probe')))
            assert 'activations' in result['probe'] and result['probe']['activations'] is not None
            if self.runner_verbose is not None and self.run_verbose(result['probe']):
                options_verbose = options.copy()
                options_verbose['activation_limit'] = result['probe']['activations']
                if szs.is_unsat(result['probe']['szs_status']):
                    options_verbose['activation_limit'] += 1
                result['verbose'] = self.result_to_record(
                    self.runner_verbose.run(problem, options_verbose, out_dir=path_join(out_dir, 'verbose')),
                    signature)
                if szs.is_unsat(result['probe']['szs_status']) and result['verbose']['szs_status'] != result['probe']['szs_status']:
                    warnings.warn(f'Unexpected SZS status in verbose run. Problem: %s. Expected (probe): %s. Actual (verbose): %s.' % (problem, result['probe']['szs_status'], result['verbose']['szs_status']))
                if (szs.is_unsat(result['probe']['szs_status']) or result['verbose']['szs_status'] == 'ACO') and result['verbose']['activations'] != result['probe']['activations']:
                    warnings.warn(f'Unexpected activations in verbose run. Problem: %s. Expected (probe): %u. Actual (verbose): %u.' % (problem, result['probe']['activations'], result['verbose']['activations']))
        return result
    
    def run_verbose(self, probe):
        if probe['activations'] <= 0:
            return False
        status = probe['szs_status']
        return status in self.szs_status_of_interest

    def weight_vector_to_dict(self, problem, weight):
        symbols = self.clausifier.clausify(problem, get_clauses=False).symbols
        if symbols is None:
            raise RuntimeError('Failed to extract signature.')
        signature = symbols.name
        assert len(weight) == len(self.clause_features) + len(signature)
        weights = {
            **dict(zip(self.clause_features, weight)),
            'symbol': dict(zip(signature, weight[len(self.clause_features):]))
        }
        assert len(weights['symbol']) == len(signature)
        return weights, symbols

    def result_to_record(self, result, signature=None):
        if result['szs_status'] in ['OSE', 'INE']:
            data = {k: result[k] for k in ['problem', 'szs_status', 'error', 'out_dir']}
            raise RuntimeError(f'Unexpected error during empirical evaluation:\n{yaml.dump(data)}')
        record = {k: result.get(k) for k in ['elapsed', 'szs_status', 'megainstructions', 'activations', 'memory', 'error']}
        if signature is not None:
            with timer() as t:
                record['clause_feature_vectors'] = proof.stdout_to_proof_samples(result['stdout'], signature,
                                                                                 self.clause_features)
            record['time_featurize'] = t.elapsed
        return record


@contextmanager
def weight_options(weights=None, functor_weight_filename=None):
    if weights is None:
        return {}
    # TODO: Set weights for other clause features.
    weight_name_to_option_name = {
        'variable_occurrence': 'variable_weight',
        'equality': 'equality_weight'
    }
    # If any of the weights is nan, we omit the respective option, effectively reverting to the default weight.
    # This happens when the respective weight has no meaning in the given problem because there are 0 occurrences of the given element.
    options = {weight_name_to_option_name[weight_name]: v for weight_name, v in weights.items() if
               weight_name != 'symbol' and not tf.math.is_nan(v)}
    with functor_weight_file(weights['symbol'], functor_weight_filename) as weights_file:
        options['functor_weight'] = os.path.abspath(weights_file.name)
        yield options


@contextmanager
def functor_weight_file(weights, filename=None):
    if filename is None:
        f = tempfile.NamedTemporaryFile('w+', suffix='.txt', prefix='vampire_functor_weight_')
    else:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = open(filename, 'w+')
    for functor, weight in weights.items():
        if functor == '=':
            # The weight of equality is passed using a dedicated option.
            # This simplifies the integration within Vampire,
            # since the equality symbol is instantiated before the problem is loaded.
            continue
        f.write(f'{to_str(weight)} {functor}\n')
    f.seek(0)
    try:
        yield f
    finally:
        f.close()


class VampireRunner:
    def __init__(self, options, run_kwargs):
        if 'include' in options and options['include'] is None:
            options = options.copy()
            del options['include']
        self.options = options
        self.run_kwargs = run_kwargs

    def run(self, problem, options=None, out_dir=None):
        cur_options = self.options
        if options is not None:
            cur_options = cur_options.copy()
            cur_options.update(options)
        return vampire.run(problem, cur_options, out_dir=out_dir, **self.run_kwargs)
