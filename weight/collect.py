import functools
import itertools
import logging
import os
import sys

import joblib
import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import save_df
from utils import subsample
from weight import tptp
from weight import vampire

log = logging.getLogger(__name__)


@hydra.main(config_path='.', config_name='config', version_base=None)
def main(cfg):
    log.info(f'cwd: {os.getcwd()}')
    log.info(f'Workspace directory: {cfg.workspace_dir}')

    workspace_abs_path = hydra.utils.to_absolute_path(cfg.workspace_dir)

    def run(problem, seed):
        log.debug(f'Attempting problem {problem} with seed {seed}')
        problem_path = tptp.problem_path(problem, cfg.tptp_path)
        out_path = os.path.join(workspace_abs_path, 'runs', problem, str(seed))
        vampire_run = functools.partial(vampire.run, vampire=cfg.vampire_cmd)
        # First run: probe, proof off
        result_probe = vampire_run(problem_path,
                                   {**cfg.options.common, 'random_seed': seed, **cfg.options.probe},
                                   os.path.join(out_path, 'probe'),
                                   **cfg.probe_run_args)
        result_verbose = None
        if result_probe['szs_status'] in ['THM', 'CAX', 'UNS', 'SAT', 'CSA']:
            # Second run: verbose, proof on
            result_verbose = vampire_run(problem_path,
                                         {**cfg.options.common, 'random_seed': seed, **cfg.options.verbose},
                                         os.path.join(out_path, 'verbose'))
        return {'probe': result_probe, 'verbose': result_verbose}

    problem_name_lists = [cfg.problem.names]
    if cfg.problem.list_file is not None:
        problem_name_lists.append(
            pd.read_csv(hydra.utils.to_absolute_path(cfg.problem.list_file), names=['problem']).problem)
    problems = sorted(set(itertools.chain.from_iterable(problem_name_lists)))

    rng = np.random.default_rng(cfg.seed)
    problems = subsample(problems, size=cfg.max_problem_count, rng=rng)
    problems = sorted(problems)

    with joblib.parallel_backend(cfg.parallel.backend, n_jobs=cfg.parallel.n_jobs):
        parallel = joblib.Parallel(verbose=cfg.parallel.verbose)
        n_total = 0
        n_successes = 0
        records = []
        if cfg.batch.count is None:
            batch_indices = itertools.count()
            disable = False
        else:
            batch_indices = range(cfg.batch.count)
            disable = cfg.batch.count <= 1
        with tqdm(batch_indices, unit='batch', desc='Collecting data', postfix={'total': 0, 'successes': 0},
                  disable=disable) as t:
            case_i = 0
            for batch_i in t:
                if cfg.batch.size is None:
                    batch_problems = rng.permutation(problems)
                else:
                    batch_problems = rng.choice(problems, cfg.batch.size)
                batch_seeds = range(case_i, case_i + len(batch_problems))
                print(f'Running {len(batch_problems)} cases', file=sys.stderr)
                results = parallel(
                    joblib.delayed(run)(problem, seed) for problem, seed in zip(batch_problems, batch_seeds))
                for problem, seed, result in zip(batch_problems, batch_seeds, results):
                    n_total += 1
                    if result['probe']['szs_status'] in ['THM', 'CAX', 'UNS', 'SAT', 'CSA']:
                        n_successes += 1
                    selected_properties = ['szs_status', 'terminationreason', 'returncode', 'elapsed', 'out_dir',
                                           'stdout_len', 'stderr_len']
                    record = {
                        'problem': problem,
                        'seed': seed,
                        'probe': {k: result['probe'][k] for k in selected_properties if k in result['probe']}
                    }
                    if result['verbose'] is not None:
                        record['verbose'] = {k: result['verbose'][k] for k in selected_properties if
                                             k in result['verbose']}
                    records.append(record)
                df = pd.json_normalize(records, sep='_')
                save_df(df, os.path.join(workspace_abs_path, 'runs'), index=False)
                case_i += len(batch_problems)
                t.set_postfix({'total': n_total, 'successes': n_successes})


if __name__ == '__main__':
    main()
