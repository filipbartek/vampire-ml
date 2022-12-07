import itertools
import logging
import os
import sys
import tempfile
import warnings
from contextlib import suppress

import joblib
import pandas as pd
import tensorflow as tf
import yaml

import questions
from utils import astype
from utils import to_str
from weight import vampire


log = logging.getLogger(__name__)


def evaluate(model, problem_names, problem_name_datasets, eval_dir, writers, clausifier, cfg, parallel, summary_prefix='empirical'):
    df = run(model, problem_names, eval_dir, clausifier, cfg, parallel)

    for dataset_name, pn in problem_name_datasets.items():
        with writers[dataset_name].as_default():
            df[f'dataset_{dataset_name}'] = df.index.isin(pn)
            cur_df = df[df.index.isin(pn)]
            tf.summary.scalar(f'{summary_prefix}/problems/total', len(cur_df))
            if len(cur_df) == 0:
                continue
            tf.summary.scalar(f'{summary_prefix}/problems/valid', cur_df.valid.sum())
            tf.summary.scalar(f'{summary_prefix}/problems/success', cur_df.success.sum())
            tf.summary.scalar(f'{summary_prefix}/problems/success_uns', cur_df.success_uns.sum())
            tf.summary.scalar(f'{summary_prefix}/problems/success_sat', cur_df.success_sat.sum())
            tf.summary.scalar(f'{summary_prefix}/success_rate', cur_df.success.mean())
            for status, count in cur_df.szs_status.value_counts(dropna=False).items():
                tf.summary.scalar(f'{summary_prefix}/problems/szs/{status}', count)
            for col in ['elapsed', 'megainstructions', 'activations']:
                if col in cur_df:
                    data = cur_df[col][cur_df.success & cur_df[col].notna()]
                    if isinstance(data.dtype, pd.core.dtypes.base.ExtensionDtype):
                        data = data.to_numpy(dtype=data.dtype.numpy_dtype)
                    tf.summary.histogram(f'{summary_prefix}/{col}', data)
            for feature in cfg.clause_features:
                with suppress(KeyError):
                    data = cur_df[f'weight_{feature}'][cur_df[f'weight_{feature}'].notna()]
                    tf.summary.histogram(f'{summary_prefix}/feature_weight/{feature}', data)
    return df


def run(model, problem_names, eval_dir, clausifier, cfg, parallel):
    log.info('Empirical evaluation...')

    model_result = None
    if model is not None and len(problem_names) > 0:
        log.info('Evaluating a model')
        # We convert problem names to Python strings.
        # They may be input as numpy strings.
        # If a list of numpy strings of length 1 is used, `tf.keras.Model.predict` is confused.
        model_result = model.predict(list(map(str, problem_names)), batch_size=cfg.batch.size)
    else:
        log.info('Evaluating baseline')

    return evaluate_options(model_result, problem_names, clausifier, cfg, cfg.options.evaluation.default,
                            parallel, out_dir=eval_dir)


def evaluate_options(model_result, problem_names, clausifier, cfg, eval_options, parallel, out_dir=None):
    options = {**cfg.options.common, **cfg.options.probe, **eval_options}

    def run_case(problem, valid, cost):
        log.debug(f'Attempting problem {problem}')
        result = {'problem': problem, 'valid': valid}
        if not valid:
            return result
        result.update(evaluate_one(problem, cost, clausifier, options, cfg, out_dir))
        log.debug(f'Attempt result:\n{yaml.dump(result)}')
        return result

    if model_result is None:
        cases = zip(problem_names, itertools.repeat(True), itertools.repeat(None))
    else:
        cases = zip(problem_names, map(bool, model_result['valid']), map(lambda x: x.numpy(), model_result['costs']))
    print(f'Running {len(problem_names)} cases', file=sys.stderr)
    results = parallel(joblib.delayed(run_case)(problem, valid, cost) for problem, valid, cost in cases)

    if cfg.per_problem_stats:
        for result in results:
            if 'weight' not in result:
                continue
            problem = result['problem']
            summary_prefix = f'problem_{problem}/feature_weight'
            for k, v in result['weight']['symbol'].items():
                if k == '=':
                    continue
                tf.summary.scalar(f'{summary_prefix}/symbol/{k}', v)
            del result['weight']['symbol']
            for k, v in result['weight'].items():
                tf.summary.scalar(f'{summary_prefix}/common/{k}', v)

    df = pd.json_normalize(results, sep='_')
    df.set_index('problem', inplace=True)
    df['success_uns'] = df.szs_status.isin(['THM', 'CAX', 'UNS'])
    df['success_sat'] = df.szs_status.isin(['SAT', 'CSA'])
    df['success'] = df.success_uns | df.success_sat
    dtypes = {'szs_status': pd.CategoricalDtype(vampire.szs.short_to_long.keys()),
              'terminationreason': 'category',
              'stdout_len': pd.UInt64Dtype(),
              'stderr_len': pd.UInt64Dtype(),
              'megainstructions': pd.UInt64Dtype(),
              'activations': pd.UInt64Dtype()}
    df = astype(df, dtypes)
    return df


def evaluate_one(problem, cost, clausifier, options, cfg, out_dir):
    log.debug(f'Attempting problem {problem}')
    result = {}
    if cost is not None:
        weight = cost
        signature = clausifier.signature(problem)
        assert len(weight) == len(cfg.clause_features) + len(signature)
        weights_common = dict(zip(cfg.clause_features, weight))
        weights = {
            **weights_common,
            'symbol': dict(zip(signature, weight[len(cfg.clause_features):]))
        }
        assert len(weights['symbol']) == len(signature)
        if cfg.per_problem_stats:
            result['weight'] = weights
        else:
            result['weight'] = weights_common
    else:
        weights = None
    # result['weights'] = weights
    problem_path = questions.config.full_problem_path(problem)
    vampire_out_dir = os.path.join(out_dir, 'problems', problem.replace('/', '_'))
    try:
        run_result = vampire_run(problem_path, options, weights, vampire=cfg.vampire_cmd,
                                 weights_filename=os.path.join(vampire_out_dir, 'functor_weight.properties'),
                                 out_dir=vampire_out_dir, **cfg.probe_run_args)
    except RuntimeError as e:
        warnings.warn(str(e))
        result['error'] = {'type': type(e), 'message': str(e)}
        return result
    selected_properties = ['szs_status', 'terminationreason', 'returncode', 'elapsed', 'out_dir',
                           'stdout_len', 'stderr_len', 'megainstructions', 'activations']
    result.update({k: run_result[k] for k in selected_properties if k in run_result})
    return result


def vampire_run(problem_path, options, weights, *args, weights_filename=None, **kwargs):
    options = options.copy()
    if 'include' in options and options['include'] is None:
        del options['include']
    weights_file = None
    if weights is not None:
        # TODO: Set weights for other clause features.
        weight_name_to_option_name = {
            'variable_occurrence': 'variable_weight',
            'equality': 'equality_weight'
        }
        options.update({weight_name_to_option_name[weight_name]: v for weight_name, v in weights.items() if
                        weight_name != 'symbol'})
        if weights_filename is None:
            weights_file = tempfile.NamedTemporaryFile('w+', suffix='.txt',
                                                       prefix=os.path.join('vampire_functor_weights_'))
        else:
            os.makedirs(os.path.dirname(weights_filename), exist_ok=True)
            weights_file = open(weights_filename, 'w+')
        for functor, weight in weights['symbol'].items():
            if functor == '=':
                # The weight of equality is passed using a dedicated option.
                # This simplifies the integration within Vampire,
                # since the equality symbol is instantiated before the problem is loaded.
                continue
            weights_file.write(f'{functor} {to_str(weight)}\n')
        weights_file.seek(0)
        options['functor_weight'] = weights_file.name
    result = vampire.run(problem_path, options, *args, **kwargs)
    if weights_file is not None:
        weights_file.close()
    return result
