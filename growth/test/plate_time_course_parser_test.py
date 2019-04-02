#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import unittest
import numpy as np
import pandas
import pylab

from growth.plate_time_course_parser import SavageLabM1000Excel
from growth.plate_spec import PlateSpec, PlateSpec384


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

    def test384Parse(self):
        parser = SavageLabM1000Excel()
        timecourse = parser.ParseFromFilename(
            'growth/data/example_growth384.xlsx')

        df = timecourse._well_df

        expected_wells = set(PlateSpec384._all_well_names())

        actual_wells = set(df['Label 1'].columns)
        self.assertTrue('time_s' in actual_wells)
        self.assertTrue('temp_C' in actual_wells)
        actual_wells.discard('time_s')
        actual_wells.discard('temp_C')
        actual_wells.discard('O2 %')
        actual_wells.discard('CO2 %')

        self.assertEqual(expected_wells, actual_wells)

        # Apply an arbitrary plate spec to a 384 well plate
        ps = PlateSpec384.FromFile('growth/plate_specs/example_plate_spec384.csv')
        blanked = timecourse.blank()
        smoothed = blanked.smooth()
        means = smoothed.mean_by_name(ps)

        # Check that the mean data has the right names - trivial check for now.
        mean_OD = means.data_for_label('Label 1')
        mean_cols = set(mean_OD.columns)
        mean_cols.discard('time_s')
        self.assertEqual(set(mean_cols), set(ps.name_to_well_mapping().keys()))


if __name__ == '__main__':
    unittest.main()
        