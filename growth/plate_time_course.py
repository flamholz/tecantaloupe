#!/usr/bin/python

from scipy import stats
import matplotlib.colors as colors
import numpy as np
import pandas

import csv
import pylab


class WellLabelMapping(dict):
    
    DEFAULT_ROWS_LABELS = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
    
    @staticmethod
    def NullMapping():
        return WellLabelMapping()
    
    @staticmethod
    def FromFile(f):
        """Assumes f is a CSV file."""
        mapping = {}
        for i, row in enumerate(csv.reader(f)):
            assert len(row) == 12, '%s != 12' % len(row)
            assert i <= len(WellLabelMapping.DEFAULT_ROWS_LABELS)
            
            row_label = WellLabelMapping.DEFAULT_ROWS_LABELS[i]
            for j, new_label in enumerate(row):
                default_label = '%s%02d' % (row_label, j+1)
                mapping[default_label] = new_label
        
        return WellLabelMapping(mapping)

    @staticmethod
    def FromFilename(filename):
        with open(filename) as f:
            return WellLabelMapping.FromFile(f)
        
    def InverseMapping(self):
        """Maps new labels to defaults in a list."""
        inverse_mapping = {}
        for orig_label, descriptive_label in self.iteritems():
            inverse_mapping.setdefault(descriptive_label, []).append(orig_label)
        return inverse_mapping
        

class PlateTimeCourse(object):
    """Immutable plate data with convenience methods for computations."""
    
    def __init__(self, well_dict):
        self._well_dict = well_dict
        self._smoothed_well_dict = None
        self._corrected_well_dict = None
        
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
                cell_label = '%s%02d' % (row_label, i+1)
                cell_data = float(cell)
                well_dict.setdefault(cell_label, []).append(cell_data)
                
        return PlateTimeCourse(well_dict)
    
    @staticmethod
    def FromFilename(fname):
        with open(fname) as f:
            return PlateTimeCourse.FromFile(f)    
    
    @property
    def well_dict(self):
        return self._well_dict
    
    @property
    def smoothed_well_dict(self):
        if self._smoothed_well_dict is not None:
            return self._smoothed_well_dict
        
        df = pandas.DataFrame(self.well_dict)
        smoothed_df = pandas.rolling_mean(df, 3)
        
        # Ensure monotonicity.
        for well_key, well_data in smoothed_df.iteritems():
            for i, value in enumerate(well_data):
                if not i:
                    continue
                
                if value < well_data[i-1]:
                    well_data[i] = well_data[i-1]
        
        
        # Second round of smoothing.
        smoother_df = pandas.rolling_mean(smoothed_df, 3)
        smoother_dict = dict((k, v.tolist()) for k,v in smoother_df.iteritems())

        self._smoothed_well_dict = smoother_dict
        return smoother_dict
    
    @property
    def corrected_smoothed_well_dict(self):
        if self._corrected_well_dict is not None:
            return self._corrected_well_dict
        
        corrected_well_dict = {}
    
        # Transforms the smoothened curve by
        # 1) subtracting the initial value which corresponds to media.
        # 2) log transforming and 3)  correcting for growth at high density 
    
        smoothed = self.smoothed_well_dict
        for well_key, well_data in smoothed.iteritems():
            corrected_well_data = []
    
            for timepoint, value in enumerate(well_data):
                # the warringer correction
                corrected_value = np.log(value + 0.8324*(value**3)) 
                corrected_well_data.append(corrected_value)
            
            corrected_well_dict[well_key] = corrected_well_data
    
        self._corrected_well_dict = corrected_well_dict
        return corrected_well_dict    
    
    def GetDoublingTimesAndLags(self, run_time, measurement_interval=30):
        """Computes the doubling times and lags.
        
        Args:
            measurement_interval: the amount of time between measurements.
                in minutes.
        
        TODO(flamholz): rethink this interface.
        """
        # Get initial values for each - a measurement prior to growth.
        # This is like having a blank, but:
        # 1) Assumes that the windowed average had a window size 3.
        # 2) Assumes that all wells were below the detection limit
        #    at the start of the experiment.
        # TODO(flamholz): do something more robust for this.
        all_init = []
        corrected_smoothed = self.corrected_smoothed_well_dict
        for well_key, well_data in corrected_smoothed.iteritems():
            all_init.append(well_data[2])
        mean_init = np.mean(all_init)

        # Maximum slope of the log-scale growth curve is specific growth rate.
        # Note: corrected dictionary is log-transformed.
        slopes_and_lags = {}
        
        for well_key, well_data in corrected_smoothed.iteritems():
            all_slopes = []
            for idx in xrange(len(well_data) - 3):
                timepoints = (idx + np.arange(4)) * measurement_interval
                data = well_data[idx:idx+4]
                regressed = stats.linregress(timepoints, data)
                all_slopes.append(regressed[:2])
        
            all_slopes.sort()
            all_slopes.reverse()
            
            ## Ignore top 2, average next 5 slopes.
            top = all_slopes[2:7]
            mean_slope, mean_intercept = np.mean(top, axis=0)
            doubling_time = np.log(2) / (mean_slope * 60)
            
            if doubling_time > run_time:
                # TODO(flamholz): maybe we want to use a sentinel here?
                # Set doubling time to None instead of run_time?
                doubling_time = run_time
            
            lag_time = (np.mean(well_data[1:3]) - (mean_intercept / mean_slope)) / 60
            if lag_time > run_time or doubling_time == run_time:
                lag_time = run_time
            
            slopes_and_lags[well_key] = (doubling_time, lag_time)
            
        return slopes_and_lags

    def PlotAll(self, label_mapping=None, measurement_interval=30):
        """Plots all the growth curves on the same figure."""
        my_label_mapping = label_mapping or {}
        fig = pylab.figure()
        for well_key, well_data in self.smoothed_well_dict.iteritems():
            n_measurements = len(well_data)
            timepoints = np.arange(n_measurements) * measurement_interval
            well_label = my_label_mapping.get(well_key, well_key)
            pylab.plot(timepoints, well_data, label=well_label, figure=fig,
                       cmap=pylab.cm.cool)
        
        pylab.xlabel('Time (Min)')
        pylab.ylabel('OD')
    
    def PlotDoublingTimeByLabels(self, label_mapping, run_time,
                                 measurement_interval=30.0):
        """Produces a boxplot of the per-label doubling times.
        
        Args:
            label_mapping: the mapping from wells to labels.
            run_time: the total running time of the experiment.
            measurement_interval: frequency of measurements, minutes.
        """
        inverse_mapping = label_mapping.InverseMapping()
        doubling_times_and_lags = self.GetDoublingTimesAndLags(
            run_time, measurement_interval)
        
        labels = sorted(inverse_mapping.keys())
        fig = pylab.figure()
        per_label_doubling_times = []
        for i, sample_label in enumerate(labels):
            cell_labels = inverse_mapping[sample_label]
            cell_doubling_times = [
                doubling_times_and_lags[l][0] for l in cell_labels]
            per_label_doubling_times.append(cell_doubling_times)
        
        pylab.boxplot(per_label_doubling_times, sym='gD')
        pylab.ylabel('Doubling Time (hours)')
        ticks = np.arange(len(labels))
        pylab.xticks(ticks, labels, rotation='35')
        pylab.ylim(0, 8)
        
    
    def PlotByLabels(self, label_mapping, measurement_interval=30.0):
        """Plots growth curves with the same labels on the same figures."""
        inverse_mapping = label_mapping.InverseMapping()
        smoothed_data = self.smoothed_well_dict
        
        for descriptive_label, orig_labels in inverse_mapping.iteritems():
            fig = pylab.figure()
            pylab.title(descriptive_label)
            pylab.xlabel('Time (Hours)')
            pylab.ylabel('OD')
            
            for label in orig_labels:
                well_data = smoothed_data[label]
                n_measurements = len(well_data)
                timepoints = np.arange(n_measurements) * measurement_interval / 60.0
                pylab.plot(timepoints, well_data, label=label, figure=fig)
                
                


from argparse import ArgumentParser
import sys

if __name__ == '__main__':
    
    parser = ArgumentParser(description='Little test script for plate data.')
    parser.add_argument('-m', '--well_label_mapping', action='store',
                        required=False,
                        help='The file with well label names.')
    parser.add_argument('data_filename', metavar='data_filename',
                        help='Plate data')
    args = parser.parse_args()
    
    well_labels = WellLabelMapping.NullMapping()
    if args.well_label_mapping:
        print 'Parsing well labels'
        well_labels = WellLabelMapping.FromFilename(args.well_label_mapping)

    print 'Filename', args.data_filename
    plate_data = PlateTimeCourse.FromFilename(args.data_filename)
    plate_data.PlotDoublingTimeByLabels(well_labels, run_time=23)
    #plate_data.PlotByLabels(well_labels)
    pylab.show()
