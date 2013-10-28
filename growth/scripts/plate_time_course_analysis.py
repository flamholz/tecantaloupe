#!/usr/bin/python

from argparse import ArgumentParser
from growth.plate_spec import PlateSpec
from growth.plate_time_course_parser import BremLabTecanParser
from growth.plate_time_course_parser import RineLabSpectramaxParser

import pylab
import sys


if __name__ == '__main__':
    
    parser = ArgumentParser(description='Little test script for plate data.')
    parser.add_argument('-i', '--measurement_interval', action='store',
                        required=False, default=30.0,
                        help='Time between measurements (minutes).')
    parser.add_argument('-p', '--plate_spec_file', action='store',
                        required=False,
                        help='The file with well label names.')
    parser.add_argument('-s', '--rine_lab_spectramax', action='store_true',
                        required=False, default=False,
                        help='Parse data from Rine lab spectramax.')
    parser.add_argument('data_filename', metavar='data_filename',
                        help='Plate data')
    args = parser.parse_args()
    
    well_labels = PlateSpec.NullMapping()
    if args.plate_spec_file:
        print 'Parsing well labels'
        well_labels = PlateSpec.FromFilename(args.plate_spec_file)

    print 'Filename', args.data_filename
    if args.rine_lab_spectramax:
        print 'Parsing as Spectramax file'
        parser = RineLabSpectramaxParser()
    else:
        print 'Parsing as Tecan file'
        parser = BremLabTecanParser(
            measurement_interval=args.measurement_interval)
        
    plate_data = parser.ParseFromFilename(args.data_filename)
    #plate_data.PlotDoublingTimeByLabels(well_labels, run_time=23)
    plate_data.PlotMeanGrowth(well_labels, include_err=True, prefixes_to_include=['41a', '42a'])
    #plate_data.PlotMeanAuc(well_labels, include_err=True, prefixes_to_include=['41a', '42a'])
    #plate_data.PrintByMeanFinalDensity(well_labels)
    pylab.show()