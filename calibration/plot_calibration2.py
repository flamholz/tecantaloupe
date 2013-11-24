#!/usr/bin/python

import csv
import pylab
import numpy as np
from scipy import stats

from growth.plate_spec import PlateSpec
from growth.plate_time_course_parser import BremLabTecanParser

parser = BremLabTecanParser(
    measurement_interval=5.5)
plate_data = parser.ParseFromFilename('calibration/data/AF_4142SerDil_3xShakeOD_Settled_19112013-002.asc')

mean_well_df = plate_data._well_df.mean()
row_labels_a = ('A', 'B', 'C', 'D')
row_labels_b = ('E', 'F', 'G', 'H')
cols = range(5, 13)

pylab.figure()
max_a = []
all_a_samples = []
for row_label in row_labels_a:
	data = []
	for col_label in cols:
		key = '%s%s' % (row_label, col_label)
		data.append(mean_well_df[key])
	
	max_a.append(np.max(data))
	all_a_samples.append(data)

max_b = []
all_b_samples = []
for row_label in row_labels_b:
	data = []
	for col_label in cols:
		key = '%s%s' % (row_label, col_label)
		data.append(mean_well_df[key])
	
	max_b.append(np.max(data))
	all_b_samples.append(data)
	
all_a_samples = np.vstack(all_a_samples)
all_b_samples = np.vstack(all_b_samples)
mean_a = np.mean(all_a_samples, axis=0)
mean_b = np.mean(all_b_samples, axis=0)
std_a = np.std(all_a_samples, axis=0)
std_b = np.std(all_b_samples, axis=0)

dilutions = np.power(2, np.arange(len(data)))
pylab.errorbar(dilutions, mean_a, yerr=std_a,
               fmt='.', color='g', label='Measured 41 dilution')
pylab.errorbar(dilutions, mean_b, yerr=std_b,
               fmt='.', color='r', label='Measured 42 dilution')

log_dilutions = np.log10(dilutions)
log_a = np.log10(mean_a)
log_b = np.log10(mean_b)
regressed_a = stats.linregress(log_dilutions, log_a)
regressed_b = stats.linregress(log_dilutions, log_b)

slope_a, intercept_a, r_a = regressed_a[:3]
predicted_log_a = slope_a * log_dilutions + intercept_a
pylab.plot(dilutions, np.power(10, predicted_log_a), 'g-',
           label='41 regression (slope=%.2f, r^2=%.2f)' % (slope_a, r_a**2))

slope_b, intercept_b, r_b = regressed_b[:3]
predicted_log_b = slope_b * log_dilutions + intercept_b
pylab.plot(dilutions, np.power(10, predicted_log_b), 'r-',
           label='42 regression (slope=%.2f, r^2=%.2f)' % (slope_b, r_b**2))

mean_max_a = np.mean(max_a)
mean_max_b = np.mean(max_b)
expected_od_a = mean_max_a / dilutions
expected_od_b = mean_max_b / dilutions

pylab.plot(dilutions, expected_od_a, 'g--', label='Expected OD for 41')
pylab.plot(dilutions, expected_od_b, 'r--', label='Expected OD for 42')

pylab.xlabel('Dilution factor')
pylab.ylabel('Tecan OD 595')
pylab.xscale('log')
pylab.yscale('log')
pylab.grid(b=True)
pylab.legend()
pylab.show()

		