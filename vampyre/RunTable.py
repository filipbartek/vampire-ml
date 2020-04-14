#!/usr/bin/env python3

import pandas as pd

from .Run import Run


class RunTable:
    def __init__(self, field_names):
        assert set(field_names) <= set(Run.fields.keys())
        self.series = {name: list() for name in field_names}

    def add_run(self, solve_run):
        for field_name, series in self.series.items():
            value = solve_run[field_name]
            series.append(value)

    def get_data_frame(self):
        typed_series = {field_name: pd.Series(series_list, dtype=Run.fields[field_name].dtype) for
                        field_name, series_list in self.series.items()}
        df = pd.DataFrame(typed_series)
        if 'output_dir' in df:
            df.set_index('output_dir', inplace=True)
            assert df.index.name == 'output_dir'
        return df

    def extend(self, runs):
        for run in runs:
            self.add_run(run)
