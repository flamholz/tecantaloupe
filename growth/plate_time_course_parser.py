#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

from growth.plate_time_course import PlateTimeCourse

import numpy as np
import pandas as pd
import re


class PlateTimeCourseParser(object):
    
    def __init__(self):
        pass
    
    def ParseFromFile(self, f):
        """Parse a file-like object.
        
        Args:
            f: the file-like object.
        
        Returns:
            A PlateTimeCourse object.
        """
        raise NotImplementedError()

    def ParseFromFilename(self, fname, sheet_name=0):
        """Convenience to parse from a filename.
        
        Opens the file, parses.
        
        Args:
            fname: the name/path to the file.
            sheet_name: the name of the
        
        Returns:
            A PlateTimeCourse object.
        """
        with open(fname, 'U') as f:
            return self.ParseFromFile(f)
 

class SavageLabM1000Excel(PlateTimeCourseParser):

    LABELS_PATTERN = re.compile('(\d+) Labels')
    SINGLE_LABEL_PATTERN = re.compile('Label: (.+)')

    def _cleanParsedDataFrame(self, df):
        """
        TODO: handle data with temp/CO2 info in it from Spark?
        """
        # this excel format puts a big header with metadata at the top.
        # we want to parse without manually futzing with the file.
        # find the start of the data
        first_col = df.columns[0]
        header_row_idx = df[df[first_col] == 'Cycle Nr.'].index[0]

        # grab the header
        colnames = df.loc[header_row_idx]
        clipped_df = df.loc[header_row_idx+1:]

        # set the column names to the right ones.
        clipped_df.columns = colnames

        # want to remove columns with no data
        # first col is timestamp - will always have some data if parsed.
        data_cols = clipped_df.columns[1:]
        empty_cols = clipped_df[data_cols].isnull().all(axis=1)
        last_index = clipped_df[empty_cols].index[0] - 1
        clipped_df = clipped_df.loc[:last_index]

        rename_mapping = {'Time [s]': 'time_s',
                          'Cycle Nr.': 'cycle_n',
                          u'Temp. [\u00b0C]': 'temp_C'}
        clipped_df.rename(columns=rename_mapping, inplace=True)
        clipped_df = clipped_df.set_index('cycle_n')
        
        # Some versions of Pandas have a default type of Object
        # for parsed files. We need to convert so we can do math.
        return clipped_df.infer_objects()

    def _splitFileToDataFrames(self, f, sheet_name=0):
        """Rather ad-hoc parsing of excel files using pandas..."""
        # first pass - count labels
        n_labels = 0
        single_label = None

        df = pd.read_excel(f, sheet_name=sheet_name)
        for row in df.index:
            l = str(df.loc[row][0])
            match = self.LABELS_PATTERN.search(l)
            if match:
                n_labels = int(match.group(1))
                break

            match = self.SINGLE_LABEL_PATTERN.search(l)
            if match:
                n_labels = 1
                single_label = match.group(1)
                break

        # second pass
        df_dict = dict()
        current_start = None
        prev_str_val = None
        current_label = None
        for row in df.index:
            val = df.loc[row][0]
            str_val = str(val).strip()

            if str_val == 'nan':
                # empty lines mean we start a new section
                if current_start and current_label:
                    current_stop = row
                    sub_df = df.loc[current_start:current_stop+1].copy()
                    sub_df = self._cleanParsedDataFrame(sub_df)
                    df_dict[current_label] = sub_df
                    current_start = None  # restart the count
            
            if str_val.startswith('Cycle Nr.'):
                current_start = row
                if n_labels == 1:
                    current_label = single_label
                else:
                    current_label = prev_str_val.strip()

            prev_str_val = str_val

        return df_dict

    def ParseFromFilename(self, f, sheet_name=0):
        """Concrete implementation.

        TODO: consistent keyword arg name for sheetname between us and pandas.
        """
        dfs = self._splitFileToDataFrames(f, sheet_name=sheet_name)
        assert dfs

        keys = sorted(dfs.keys())
        ordered_dfs = [dfs[k] for k in keys]

        merged_df = pd.concat(
            ordered_dfs, axis=1, keys=keys,
            names=['measurement_type', 'well'])
        return PlateTimeCourse(merged_df)


class CoatesLabSunriseExcel(PlateTimeCourseParser):
    """Assumes you exported wells along the columns with a timestamp.

    Can only measure absorbance.
    """
    COLS = list(map(str, np.arange(1, 13)))
    ROWS = 'A,B,C,D,E,F,G,H'.split(',')
    LABEL = 'OD600'

    def ParseFromFilename(self, f, sheet_name=0):
        """Concrete implementation."""
        wells = ['%s%s' % (r, c) for c in self.COLS
                 for r in self.ROWS]
        h = ['time_s'] + wells
        df = pd.read_excel(f, names=h, sheetname=sheet_name)
        last_id = df.index[-1]
        df.drop(axis=0, labels=last_id, inplace=True)

        time_s = [int(t.strip('s')) for t in df.time_s.values]
        df.time_s = time_s

        merged_df = pd.concat(
            [df], axis=1, keys=[self.LABEL],
            names=['measurement_type', 'well'])

        return PlateTimeCourse(merged_df)
