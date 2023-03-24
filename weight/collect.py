import functools
import logging
import os
import sys

import joblib
import hydra
import numpy as np
import omegaconf
import pandas as pd

from utils import get_problems
from utils import save_df
from utils import save_list
from weight import tptp
from weight import vampire

log = logging.getLogger(__name__)


@hydra.main(config_path='.', config_name='config', version_base=None)
def main(cfg):
    log.info(f'cwd: {os.getcwd()}')
    log.info(f'Workspace directory: {cfg.workspace_dir}')

    workspace_abs_path = hydra.utils.to_absolute_path(cfg.workspace_dir)

    rng = np.random.default_rng(cfg.seed)
    problems, problem_root_dir = get_problems(cfg.problem, rng)
    log.info(f'Problem common path: {problem_root_dir}')
    save_list(problems, os.path.join(workspace_abs_path, 'problems.txt'))

    def run(problem, seed):
        log.debug(f'Attempting problem {problem} with seed {seed}')
        try:
            problem_path = tptp.problem_path(problem, cfg.tptp_path)
        except omegaconf.errors.InterpolationResolutionError:
            problem_path = problem
        problem_name = os.path.relpath(problem, problem_root_dir)
        out_path = os.path.join(workspace_abs_path, 'runs', problem_name, str(seed))
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

    with joblib.parallel_backend(cfg.parallel.backend, n_jobs=cfg.parallel.n_jobs):
        batch_problems = problems
        batch_seeds = range(len(batch_problems))
        print(f'Running {len(batch_problems)} cases', file=sys.stderr)
        results = joblib.Parallel(verbose=cfg.parallel.verbose)(
            joblib.delayed(run)(problem, seed) for problem, seed in zip(batch_problems, batch_seeds))
        records = []
        for problem, seed, result in zip(batch_problems, batch_seeds, results):
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

        log.info(f'Number of verbose runs: {(~df.verbose_out_dir.isna()).sum()}')
        log.info(f'SZS status distribution: {df.probe_szs_status.value_counts()}')


if __name__ == '__main__':
    main()
