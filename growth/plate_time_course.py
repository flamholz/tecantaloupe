#!/usr/bin/python

import numpy as np
import matplotlib.colors as colors

import pylab


class PlateTimeCourse(object):

    def __init__(self, well_dict):
        self.well_dict = well_dict

    @staticmethod
    def FromFile(f):
        well_dict = {}
        for line in f:
            if line.startswith('<>'):
                continue  # Skip header lines.
            
            row_data = line.strip().split("\t")
            row_label = row_data[0]
            
            # First entry in line is the row label.
            for i, cell in enumerate(row_data[1:]):
                cell_label = '%s%02d' % (row_label, i)
                cell_data = float(cell)
                well_dict.setdefault(cell_label, []).append(cell_data)
                
        return PlateTimeCourse(well_dict)
    
    @staticmethod
    def FromFilename(fname):
        with open(fname) as f:
            return PlateTimeCourse.FromFile(f)
        
    
    @property
    def smoothed_well_dict(self):
        smooth_dict = {}
        # First round of smoothing - rolling average.
        # TODO(flamholz): do this with pandas?
        for well_key, well_data in self.well_dict.iteritems():
            n_values = len(well_data)
            smooth_values = np.zeros(n_values)
            for i, value in enumerate(well_data[1:-1]):
                real_idx = i + 1
                prev_value = well_data[i]
                cur_value = well_data[i]
                next_value = well_data[i+2]
                smooth_values[real_idx] = np.mean((prev_value, cur_value, next_value))
                smooth_dict[well_key] = smooth_values.tolist()
        
        # Ensure monotonicity.
        for well_data in smooth_dict.itervalues():
            for i, value in enumerate(well_data):
                if not i:
                    continue
                
                if value < well_data[i-1]:
                    well_data[i] = well_data[i-1]
        
        # Second round of smoothing.
        # TODO(flamholz): definitely do this with a library method.
        smoother_dict = {}
        for well_key, well_data in smooth_dict.iteritems():
            n_values = len(well_data)
            smoother_values = np.zeros(n_values)
            for i, value in enumerate(well_data[1:-1]):
                real_idx = i + 1
                prev_value = well_data[i]
                cur_value = well_data[i]
                next_value = well_data[i+2]
                smoother_values[real_idx] = np.mean((prev_value, cur_value, next_value))
                smoother_dict[well_key] = smoother_values.tolist()
        
        return smoother_dict
    
    def plot(self, measurement_interval=30):
        fig = pylab.figure()
        colors.Colormap('bone')
        for well_key, well_data in self.smoothed_well_dict.iteritems():
            n_measurements = len(well_data)
            timepoints = np.arange(n_measurements) * measurement_interval
            pylab.plot(timepoints, well_data, label=well_key, figure=fig)
        
        pylab.xlabel('Time (Min)')
        pylab.ylabel('OD')


import sys

if __name__ == '__main__':
    filename = sys.argv[1]
    print 'Filename', filename
    plate_data = PlateTimeCourse.FromFilename(filename)
    plate_data.plot()
    pylab.show()
