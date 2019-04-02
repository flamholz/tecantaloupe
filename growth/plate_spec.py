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


    def __init__(self, df):
        """Initialize with a DataFrame describing the plate.
    
        Args:
            df: Pandas DataFrame. See
                plate_specs/example_plate_spec.csv
                for format.
        """
        self.df = df
        self.cols = self.COLS
        self.rows = self.ROWS

    @classmethod
    def _all_wells(cls):
        return tuple(itertools.product(cls.ROWS, cls.COLS))

    @classmethod
    def _all_well_names(cls):
        return tuple('%s%s' % (r, c) for r, c in cls._all_wells())

    def well_to_name_mapping(self):
        """Returns a mapping from cells -> name."""
        mapping = dict()
        for row, col in self._all_wells():
            s = '%s%s' % (row, col)
            n = self.df.name[col][row]
            mapping[s] = n
        return mapping

    def name_to_well_mapping(self):
        """Returns a mapping from name -> cells."""
        mapping = dict()
        for row, col in self._all_wells():
            s = '%s%s' % (row, col)
            n = self.df.name[col][row]
            mapping.setdefault(n, []).append(s)
        return mapping

    @classmethod
    def NullPlateSpec(cls):
        """
        Returns an empty PlateSpec in the right format for 96 well plates.
        """
        rows = cls.ROWS
        cols = cls.COLS

        tuples = [('name', v) for v in cols]

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
        return cls(df)

    @classmethod
    def FromFile(cls, f):
        """Assumes f is a CSV file.

        Args:
            f: file handle or path to read from.
                Better be in the right format.
        """
        df = pd.read_csv(f, header=[0, 1], index_col=[0])
        return cls(df)


class PlateSpec384(PlateSpec):
    """Read/write specifications for 384 well plates.

    TODO: make this generic for any plate size.
    """
    COLS = list(map(str, np.arange(1, 25)))
    ROWS = 'A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P'.split(',')
    
    def __init__(self, df):
        super().__init__(df)

        self.cols = PlateSpec384.COLS
        self.rows = PlateSpec384.ROWS