#!/usr/bin/python

import csv
import pylab
import numpy as np
from scipy import stats

from growth.plate_spec import PlateSpec
from growth.plate_time_course_parser import BremLabTecanParser

parser = BremLabTecanParser(
    measurement_interval=5.5)
plate_data = parser.ParseFromFilename('calibration/data/Avi_41_42_Overnight_18112013-001.asc')

tecan_ods = [1.4494, 1.4401999999999999, 1.5142, 1.4943]
mixed_spec_ods = np.array([0.97, 0.93, 1.03, 0.96]) * 20.0

print 'Tecan OD std', np.std(tecan_ods)
print 'Spec OD std', np.std(mixed_spec_ods)


pylab.title('Spec OD vs. Tecan OD')
pylab.xlabel('Spec ODs')
pylab.ylabel('Tecan ODs')
pylab.plot(mixed_spec_ods, tecan_ods, 'o')
pylab.show()