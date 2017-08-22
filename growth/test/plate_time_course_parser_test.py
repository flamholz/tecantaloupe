#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import unittest
import numpy as np
import pandas
import pylab

from growth.plate_time_course_parser import SavageLabM1000Excel


class PlateTimecourseParserTest(unittest.TestCase):
    
    def testSimpleParse(self):
        # parsing a dataset with one measurement.
        parser = SavageLabM1000Excel()
        timecourse = parser.ParseFromFilename(
            'growth/data/csosCA_inductiontest_08202017.xlsx')

        df = timecourse._well_df
        print df
        print df.columns
        data_cols = df.columns[2:]
        self.assertEquals('Time [s]', df.columns[0])
        self.assertTrue(df.columns[1].startswith('Temp'))

    def testMultipleParse(self):
        parser = SavageLabM1000Excel()
        timecourse = parser.ParseFromFilename(
            'growth/data/Emeric05_12_17 C2C2 run 1.xlsx')

        df = timecourse._well_df
        #print df
        #print df.columns


if __name__ == '__main__':
    unittest.main()
        