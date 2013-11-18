#!/usr/bin/python

import csv
import pylab
import numpy as np

condition_names = ['Replicate 1', 'Replicate 2',
                   'Replicate 3', 'Replicate 4']
dilutions = []
ods = []

with open('calibration/data/AF_SerDil_10sShake.csv') as f:
	reader = csv.DictReader(f)
	for row in reader:
		dilutions.append(float(row['Dilution']))
		
		ods_for_dilution = []
		for cname in condition_names:
			ods_for_dilution.append(float(row[cname]))
		ods.append(ods_for_dilution)

# Make things into arrays. Calculate means.
dilutions = np.array(dilutions)
ods = np.vstack(ods)
mean_ods = np.mean(ods, axis=1)
std_ods = np.std(ods, axis=1) / np.sqrt(len(condition_names))

# Simple linear model diluting from average max measurement
mean_max_od = np.mean(ods[0,:])
expected_ods = mean_max_od * np.ones(dilutions.size)
expected_ods /= dilutions

# Simple linear model dilution spec measurement.
spec_od = 15.6
expected_spec_ods = spec_od * np.ones(dilutions.size)
expected_spec_ods /= dilutions

# Range for plot
min_dilution = np.min(dilutions)
max_dilution = np.max(dilutions)
min_od = min(np.min(mean_ods), np.min(expected_ods), np.min(expected_spec_ods))
max_od = max(np.max(mean_ods), np.max(expected_ods), np.max(expected_spec_ods))

pylab.figure()
pylab.grid(b=True)
pylab.xlabel('Dilution factor')
pylab.ylabel('Tecan OD 595')
pylab.xscale('log')
pylab.yscale('log')
pylab.xlim((min_dilution / 2, max_dilution * 2))
pylab.ylim((min_od / 2, max_od * 2))

pylab.errorbar(dilutions, mean_ods, yerr=std_ods, fmt='.',
               label='Measured Tecan OD')
pylab.plot(dilutions, expected_ods, '--', label='Expected Tecan OD')
pylab.plot(dilutions, expected_spec_ods, label='Expected Spec OD')
pylab.legend()
pylab.show()
