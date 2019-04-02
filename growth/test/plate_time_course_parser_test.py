#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import unittest
import numpy as np
import pandas
import pylab

from growth.plate_time_course_parser import SavageLabM1000Excel
from growth.plate_spec import PlateSpec


class PlateTimecourseParserTest(unittest.TestCase):
    
    def testSimpleParse(self):
        # parsing a dataset with one measurement.
        parser = SavageLabM1000Excel()
        timecourse = parser.ParseFromFilename(
            'growth/data/example_growth.xlsx')

        df = timecourse._well_df

        expected_wells = set(PlateSpec._all_well_names())
        actual_wells = set(df['abs600'].columns)
        self.assertTrue('time_s' in actual_wells)
        self.assertTrue('temp_C' in actual_wells)
        actual_wells.discard('time_s')
        actual_wells.discard('temp_C')

        self.assertEqual(expected_wells, actual_wells)

    def testMultipleParse(self):
        parser = SavageLabM1000Excel()
        timecourse = parser.ParseFromFilename(
            'growth/data/example_data_multimeasurement.xlsx')

        df = timecourse._well_df

        expected_wells = set(PlateSpec._all_well_names())
        actual_wells = set(df['GFP'].columns)
        self.assertTrue('time_s' in actual_wells)
        self.assertTrue('temp_C' in actual_wells)
        actual_wells.discard('time_s')
        actual_wells.discard('temp_C')

        self.assertEqual(expected_wells, actual_wells)

if __name__ == '__main__':
    unittest.main()
        