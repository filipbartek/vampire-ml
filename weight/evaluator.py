import json
import os
import sys
import tempfile
from contextlib import contextmanager

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

from questions.memory import memory
from questions.utils import timer
from utils import error_record
from utils import path_join
from utils import to_str
from vampire import szs
from weight import dataset
from weight import proof
from weight import vampire


class Evaluator:
    def evaluate(self, problem_weights, out_dir=None, iteration=None, parallel=None):
        if parallel is None:
            parallel = joblib.Parallel()

        def evaluate_one(problem, i, weight):
            cur_out_dir = out_dir
            if cur_out_dir is not None:
                if len(problem_weights) > 1:
                    cur_out_dir = os.path.join(cur_out_dir, problem.replace('/', '_'))
                if len(problem_weights[problem]) > 1:
                    cur_out_dir = os.path.join(cur_out_dir, str(i))
            return {
                'problem': problem,
                'weight_idx': i,
                'weight': weight,
                'out_dir': cur_out_dir,
                **self.evaluate_one(problem, weight, out_dir=cur_out_dir,
                                    iteration=os.path.join(self.get_problem_basename(problem), iteration))
            }

        def gen_cases():
            for problem, weight_matrix in problem_weights.items():
                if isinstance(weight_matrix, dict):
                    weight_arrays = weight_matrix.items()
                else:
                    weight_arrays = enumerate(weight_matrix)
                for i, weight_array in weight_arrays:
                    yield problem, i, weight_array

        cases = list(gen_cases())

        print(f'Evaluating {len(cases)} weight vectors on {len(problem_weights)} problems', file=sys.stderr)
        return parallel(joblib.delayed(evaluate_one)(problem, i, weight) for problem, i, weight in cases)

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


def may_be_saturation_based_search(status):
    return szs.is_unsat(status) or status in ['TMO', 'MMO', 'GUP', 'INC', 'ACO', 'INO']


@memory.cache(ignore=['out_dir'])
def empirical_evaluate_one(evaluator, problem, weight, out_dir):
    weights_dict, symbols = evaluator.weight_vector_to_dict(problem, weight)
    signature = symbols.name.tolist()
    result = {}
    with weight_options(weights_dict, path_join(out_dir, 'functor_weight.txt')) as options:
        result['probe'] = evaluator.result_to_record(
            evaluator.runner_probe.run(problem, options, out_dir=path_join(out_dir, 'probe')))
        assert 'activations' in result['probe'] and result['probe']['activations'] is not None
        if evaluator.runner_verbose is not None and may_be_saturation_based_search(result['probe']['szs_status']) and result['probe']['activations'] > 0:
            options_verbose = options.copy()
            options_verbose['activation_limit'] = result['probe']['activations']
            if szs.is_unsat(result['probe']['szs_status']):
                options_verbose['activation_limit'] += 1
            result['verbose'] = evaluator.result_to_record(
                evaluator.runner_verbose.run(problem, options_verbose, out_dir=path_join(out_dir, 'verbose')),
                signature)
            assert not szs.is_unsat(result['probe']['szs_status']) or (
                    result['verbose']['szs_status'] == result['probe']['szs_status'] and result['verbose'][
                'activations'] == result['probe']['activations'])
            assert result['verbose']['szs_status'] != 'ACO' or result['verbose']['activations'] == result['probe'][
                'activations']
    return result


class Empirical(Evaluator):
    def __init__(self, runner_probe, clausifier, clause_features, clause_max_len=None, clause_max_terminals=None,
                 runner_verbose=None, plot_max_features=0):
        self.runner_probe = runner_probe
        self.runner_verbose = runner_verbose
        self.clausifier = clausifier
        self.clause_features = clause_features
        self.clause_max_len = clause_max_len
        self.clause_max_terminals = clause_max_terminals
        self.plot_max_features = plot_max_features

    def evaluate_one(self, problem, weight, out_dir=None, iteration=None):
        result = {}
        if iteration is not None and len(weight) <= self.plot_max_features:
            weights_dict, symbols = self.weight_vector_to_dict(problem, weight)
            with timer() as t:
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
        if result['probe']['szs_status'] == 'OSE':
            raise RuntimeError(result['probe']['error'])
        return result

    def weight_vector_to_dict(self, problem, weight):
        symbols = self.clausifier.clausify(problem, get_clauses=False).symbols
        signature = symbols.name
        assert len(weight) == len(self.clause_features) + len(signature)
        weights = {
            **dict(zip(self.clause_features, weight)),
            'symbol': dict(zip(signature, weight[len(self.clause_features):]))
        }
        assert len(weights['symbol']) == len(signature)
        return weights, symbols

    def result_to_record(self, result, signature=None):
        record = {k: result.get(k) for k in ['elapsed', 'szs_status', 'megainstructions', 'activations', 'memory']}
        if signature is not None:
            with timer() as t:
                record['clause_feature_vectors'] = proof.stdout_to_proof_samples(result['stdout'], signature,
                                                                                 self.clause_features,
                                                                                 clause_max_len=self.clause_max_len,
                                                                                 clause_max_terminals=self.clause_max_terminals)
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
    options = {weight_name_to_option_name[weight_name]: v for weight_name, v in weights.items() if
               weight_name != 'symbol'}
    with functor_weight_file(weights['symbol'], functor_weight_filename) as weights_file:
        options['functor_weight'] = weights_file.name
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
        f.write(f'{functor} {to_str(weight)}\n')
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
