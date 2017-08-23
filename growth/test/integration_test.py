#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import unittest
import numpy as np
import pandas
import pylab

from growth.plate_spec import PlateSpec
from growth.plate_time_course_parser import SavageLabM1000Excel


class IntegrationTest(unittest.TestCase):
    
    def testSimpleParse(self):
        # parsing a dataset with one measurement.
        ps = PlateSpec.FromFile('growth/plate_specs/example_plate_spec.csv')

        parser = SavageLabM1000Excel()
        timecourse = parser.ParseFromFilename(
            'growth/data/example_data.xlsx')

        blanked = timecourse.blank()
        smoothed = timecourse.smooth()
        smoothed_OD = smoothed.data_for_label('abs600')
        # 96 wells, time column
        self.assertEquals(97, len(smoothed_OD.columns))

        means = timecourse.mean_by_name(ps)
        sems = timecourse.sem_by_name(ps)

        mean_OD = means.data_for_label('abs600')
        # 33 unique measurements.
        self.assertEquals(34, len(mean_OD.columns))
        self.assertTrue('time_s' in mean_OD.columns)

    def testMultiMeasurementParse(self):
        # parsing a dataset with one measurement.
        ps = PlateSpec.FromFile(
            'growth/plate_specs/example_plate_spec_multimeasurement.csv')

        parser = SavageLabM1000Excel()
        timecourse = parser.ParseFromFilename(
            'growth/data/example_data_multimeasurement.xlsx')

        blanked = timecourse.blank()
        smoothed = timecourse.smooth()
        smoothed_OD = smoothed.data_for_label('abs600')
        # 96 wells, time column
        self.assertEquals(97, smoothed_OD.columns.size)

        means = timecourse.mean_by_name(ps)
        sems = timecourse.sem_by_name(ps)

        mean_OD = means.data_for_label('abs600')
        # 32 unique measurements, time column
        self.assertEquals(33, mean_OD.columns.size)
        self.assertTrue('time_s' in mean_OD.columns)

        GFP_per_OD = means.ratio_time_course('GFP', 'abs600')
        ratio_data = GFP_per_OD.data_for_label('GFP/abs600')
        self.assertEquals(33, ratio_data.columns.size)

if __name__ == '__main__':
    unittest.main()
        