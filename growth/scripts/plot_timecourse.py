#!/usr/bin/python

from argparse import ArgumentParser
from growth.plate_spec import PlateSpec
from growth.plate_time_course_parser import SavageLabM1000Excel
from matplotlib import pyplot as plt

import numpy as np
import seaborn
import sys


if __name__ == '__main__':

    parser = ArgumentParser(description='Little test script for plate data.')
    parser.add_argument('-p', '--plate_spec_file', action='store',
                        required=False,
                        help='The file with well label names.')
    parser.add_argument('-m', '--measurement_type', action='store',
                        required=False,
                        help='The measurement type to plot.')
    parser.add_argument('data_filename', metavar='data_filename',
                        help='Plate data')
    args = parser.parse_args()

    ps = PlateSpec.NullPlateSpec()
    if args.plate_spec_file:
        print 'Parsing well labels'
        ps = PlateSpec.FromFile(args.plate_spec_file)

    print 'Filename', args.data_filename
    parser = SavageLabM1000Excel()
    timecourse = parser.ParseFromFilename(args.data_filename)

    blanked = timecourse.blank()
    smoothed = blanked.smooth()

    means = smoothed.mean_by_name(ps)
    sems = smoothed.sem_by_name(ps)

    seaborn.set_style('white')
    if args.measurement_type:
        mtype = args.measurement_type
        print 'Plotting', mtype, 'only'
        means[mtype].plot(
            yerr=sems[mtype], figsize=(20, 10),
            title=mtype)
    else:
        means.plot(
            yerr=sems, figsize=(20, 10))
    plt.show()
