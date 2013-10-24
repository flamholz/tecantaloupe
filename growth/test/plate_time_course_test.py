#!/usr/bin/python

import unittest
import numpy as np
import pandas
import pylab

from growth.plate_time_course import PlateTimeCourse


class GrowthRatesTest(unittest.TestCase):
    
    def GenFakeData(self,
                    doubling_time_hrs,
                    initial_amount,
                    detection_limit,
                    measurement_interval_mins,
                    n_measurements,
                    add_noise=False):
        # 30 minute measurement intervals
        timepoints = np.arange(n_measurements) * float(measurement_interval_mins)
        doubling_time_mins = doubling_time_hrs * 60
        
        data = initial_amount * np.exp2(timepoints / doubling_time_mins)
        beneath_limit = np.where(data < detection_limit)
        data[beneath_limit] = detection_limit
        
        if add_noise:
            data += np.random.normal(0, initial_amount, n_measurements)
        
        data_series = pandas.Series(data, index=timepoints)
        df = pandas.DataFrame({'my_strain': data_series})
        return df
    
    def testDoublingTime(self):
        detection_limit = 0.2
        initial_amt = 0.02
        doubling_time_hrs = 1.35
        measurement_interval_mins = 30.0
        n_measurements = 80.0
        run_time = measurement_interval_mins * n_measurements / 60.0
        min_reading = 4.0
        
        df = self.GenFakeData(doubling_time_hrs,
                              initial_amt,
                              detection_limit,
                              measurement_interval_mins,
                              n_measurements)
        tc = PlateTimeCourse(df)
        dt_dict = tc.GetDoublingTimes(run_time, measurement_interval_mins,
                                      min_reading=min_reading)
        inferred_dt = dt_dict['my_strain']
        self.assertAlmostEqual(doubling_time_hrs, inferred_dt, 1)

    def testDoublingTimeWithNoise(self):
        detection_limit = 0.2
        initial_amt = 0.02
        doubling_time_hrs = 1.35
        measurement_interval_mins = 30.0
        n_measurements = 80.0
        run_time = measurement_interval_mins * n_measurements / 60.0
        min_reading = 4.0
        
        df = self.GenFakeData(doubling_time_hrs,
                              initial_amt,
                              detection_limit,
                              measurement_interval_mins,
                              n_measurements,
                              add_noise=True)
                
        tc = PlateTimeCourse(df)
        dt_dict = tc.GetDoublingTimes(run_time, measurement_interval_mins,
                                      min_reading=min_reading)
        inferred_dt = dt_dict['my_strain']
        self.assertAlmostEqual(doubling_time_hrs, inferred_dt, 1)

if __name__ == '__main__':
    unittest.main()
        
        