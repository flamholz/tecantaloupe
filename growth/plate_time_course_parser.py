#!/usr/bin/python

from growth.plate_time_course import PlateTimeCourse

import numpy as np
import pandas as pd


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

    def ParseFromFilename(self, fname):
        """Convenience to parse from a filename.
        
        Opens the file, parses.
        
        Args:
            fname: the name/path to the file.
        
        Returns:
            A PlateTimeCourse object.
        """
        with open(fname, 'U') as f:
            return self.ParseFromFile(f)
 

class SavageLabM1000Excel(PlateTimeCourseParser):

    def ParseFromFilename(self, f):
        """Concrete implementation."""
        df = pd.read_excel(f)

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
        clipped_df = clipped_df.set_index('Cycle Nr.')
        #clipped_df.columns[0] = 'time_s'
        #clipped_df.columns[1] = 'temp_C'

        return PlateTimeCourse(clipped_df)
   