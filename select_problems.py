#!/usr/bin/env python3.7

import logging
import os

import numpy as np
import pandas as pd

from vampire_ml.results import save_df

if __name__ == '__main__':
    np.seterr(all='raise')
    logging.basicConfig(level=logging.INFO)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Problems dataframe pickle')
    parser.add_argument('--output-dir')
    parser.add_argument('--n', default=100, type=int, help='Maximum number of problems to output')
    parser.add_argument('--saturation-iterations-nunique-min', default=12, type=int)
    namespace = parser.parse_args()

    # Load
    problems = pd.read_pickle(namespace.input)

    # Select and sort
    problems_chosen = problems[
        problems[('saturation_iterations', 'nunique')] >= namespace.saturation_iterations_nunique_min]
    problems_chosen = problems_chosen.sort_values(('score', 'std'), ascending=False)
    problems_chosen = problems_chosen[:namespace.n]

    # Save
    save_df(problems_chosen, 'problems_selected', namespace.output_dir)
    path_txt = 'problems_selected.txt'
    if namespace.output_dir is not None:
        path_txt = os.path.join(namespace.output_dir, path_txt)
    problems_chosen.index.to_series().to_csv(path_txt, index=False, header=False)
