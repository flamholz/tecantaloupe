#!/usr/bin/python

import pandas as pd
import numpy as np
import itertools


class PlateSpec(dict):
    """Read/write specifications for 96 well plates.

    TODO: make this generic for any plate size.
    """
    
    COLS = list(map(str, np.arange(1, 13)))
    ROWS = 'A,B,C,D,E,F,G,H'.split(',')
    ALL_WELLS = tuple(itertools.product(ROWS, COLS))
    ALL_WELL_NAMES = tuple('%s%s' % (r, c) for r, c in ALL_WELLS)

    def __init__(self, df):
        """Initialize with a DataFrame describing the plate.
    
        Args:
            df: Pandas DataFrame. See
                plate_specs/example_plate_spec.csv
                for format.
        """
        self.df = df

    def well_to_name_mapping(self):
        """Returns a mapping from cells -> name."""
        mapping = dict()
        for row, col in PlateSpec.ALL_WELLS:
            s = '%s%s' % (row, col)
            n = self.df.name[col][row]
            mapping[s] = n
        return mapping

    def name_to_well_mapping(self):
        """Returns a mapping from name -> cells."""
        mapping = dict()
        for row, col in PlateSpec.ALL_WELLS:
            s = '%s%s' % (row, col)
            n = self.df.name[col][row]
            mapping.setdefault(n, []).append(s)
        return mapping

    @staticmethod
    def NullPlateSpec():
        """
        Returns an empty PlateSpec in the right format for 96 well plates.
        """
        rows = PlateSpec.ROWS
        cols = PlateSpec.COLS

        arrays = [['name'], cols]
        tuples = list(PlateSpec.ALL_WELLS)

        index = pd.MultiIndex.from_tuples(
            tuples, names=['value_type', 'column'])
        well_names = []
        for row in rows:
            row_data = []
            for col in cols:
                s = '%s%s' % (row, col)
                row_data.append(s)
            well_names.append(row_data)

        df = pd.DataFrame(well_names, index=rows, columns=index)
        return PlateSpec(df)

    @staticmethod
    def FromFile(f):
        """Assumes f is a CSV file.

        Args:
            f: file handle or path to read from.
                Better be in the right format.
        """
        df = pd.read_csv(f, header=[0, 1], index_col=[0])
        return PlateSpec(df)
