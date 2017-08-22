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

    figure = plt.figure(figsize=(20, 10))
    seaborn.set_style('white')
    colors = seaborn.color_palette()
    time_h = smoothed._well_df['Time [s]'] / (60.0*60.0)

    for i, name in enumerate(means.columns):
        color = colors[i % 4]
        ls = '--'

        plt.errorbar(time_h, means[name], yerr=sems[name],
                     ls=ls, color=color, label=name)

    plt.legend(loc='best', fontsize=8)
    plt.xlim(0, 24.1)
    plt.xticks(np.arange(0, 24.1, 3), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Time (Hours)', fontsize=16)
    plt.ylabel('OD600', fontsize=16)

    plt.show()