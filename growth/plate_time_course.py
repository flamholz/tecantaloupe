#!/usr/bin/python

from scipy import stats
import csv
import matplotlib.colors as colors
import numpy as np
import pandas
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
    
    def __init__(self, well_data_frame):
        self._well_df = well_data_frame
        self._smoothed_well_df = None
        self._corrected_well_df = None
        
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
                
        return PlateTimeCourse(pandas.DataFrame(well_dict))
    
    @staticmethod
    def FromFilename(fname):
        with open(fname) as f:
            return PlateTimeCourse.FromFile(f)
    
    @property
    def smoothed_well_df(self):
        if self._smoothed_well_df is not None:
            return self._smoothed_well_df
        
        smoothed_df = pandas.rolling_mean(self._well_df, 3)
        
        # Ensure monotonicity.
        for well_key, well_data in smoothed_df.iteritems():
            for i, value in enumerate(well_data):
                if not i:
                    continue
                
                if value < well_data[i-1]:
                    well_data[i] = well_data[i-1]
        
        
        # Second round of smoothing.
        self._smoothed_well_df = pandas.rolling_mean(smoothed_df, 3)
        return self._smoothed_well_df
    
    @property
    def corrected_log_smoothed_well_df(self):
        if self._corrected_well_df is not None:
            return self._corrected_well_df
            
        # Transforms the smoothened curve by
        # 1) subtracting the initial value which corresponds to media.
        # 2) log transforming and 3)  correcting for growth at high density 
    
        smoothed_df = self.smoothed_well_df
        sorted_df = smoothed_df.sort()
        mean_init_vals = sorted_df[2:5].mean()
        corrected_df = smoothed_df - mean_init_vals
        
        # Extra term is the Warringer correction
        # See PMID 21698134
        corrected_df = np.log(corrected_df + 0.8324*(smoothed_df**3))
        self._corrected_well_df = corrected_df
        return corrected_df
    
    def GetDoublingTimes(self, run_time, measurement_interval=30):
        """Computes the doubling times and lags.
        
        Args:
            measurement_interval: the amount of time between measurements.
                in minutes.
        
        TODO(flamholz): rethink this interface.
        """
        # Maximum slope of the log-scale growth curve is specific growth rate.
        # Note: corrected dictionary is log-transformed already.
        corrected_smoothed = self.corrected_log_smoothed_well_df
        doubling_times = {}
        
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
            
            doubling_times[well_key] = doubling_time
        
        return doubling_times

    def PlotAll(self, label_mapping=None, measurement_interval=30):
        """Plots all the growth curves on the same figure."""
        my_label_mapping = label_mapping or {}
        fig = pylab.figure()
        for well_key, well_data in self.smoothed_well_df.iteritems():
            n_measurements = len(well_data)
            timepoints = np.arange(n_measurements) * measurement_interval
            well_label = my_label_mapping.get(well_key, well_key)
            pylab.plot(timepoints, well_data, label=well_label, figure=fig)
        
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
        doubling_times_and_lags = self.GetDoublingTimes(
            run_time, measurement_interval)
        
        labels = sorted(inverse_mapping.keys())
        fig = pylab.figure()
        per_label_doubling_times = []
        for i, sample_label in enumerate(labels):
            cell_labels = inverse_mapping[sample_label]
            cell_doubling_times = [
                doubling_times_and_lags[l] for l in cell_labels]
            per_label_doubling_times.append(cell_doubling_times)
        
        pylab.boxplot(per_label_doubling_times, sym='gD')
        pylab.ylabel('Doubling Time (hours)')
        ticks = np.arange(len(labels))
        pylab.xticks(ticks, labels, rotation='35')
        pylab.ylim(0, 4)
        
    
    def PlotByLabels(self, label_mapping, measurement_interval=30.0):
        """Plots growth curves with the same labels on the same figures."""
        inverse_mapping = label_mapping.InverseMapping()
        smoothed_data = self.smoothed_well_df
        
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
    plate_data.PlotByLabels(well_labels)
    pylab.show()
