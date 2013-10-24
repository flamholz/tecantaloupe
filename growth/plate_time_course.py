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
    def FromFile(f, measurement_interval=30.0):
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
        
        n_measurements = len(well_dict.values()[0])
        timepoints = np.arange(float(n_measurements)) * measurement_interval

        return PlateTimeCourse(pandas.DataFrame(well_dict, index=timepoints))
    
    @staticmethod
    def FromFilename(fname, measurement_interval=30.0):
        with open(fname) as f:
            return PlateTimeCourse.FromFile(f, measurement_interval)
    
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
        #corrected_df = np.log(corrected_df + 0.8324*(corrected_df**3))
        # TODO(flamholz): think on this...
        
        corrected_df = np.log(corrected_df)
        self._corrected_well_df = corrected_df
        return corrected_df
    
    def GetDoublingTimes(self, run_time,
                         measurement_interval=30,
                         min_reading=0.05):
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
            all_regressions = []
            # For each 4-measurement windows.
            # 1) discard if minimum value beneath user-defined limit.
            # 2) regress against time.
            # 3) keep regression slope.
            for idx in xrange(len(well_data) - 3):
                timepoints = (idx + np.arange(4)) * measurement_interval
                data = well_data[idx:idx+4]
                
                if np.min(data) < np.log(min_reading):
                    continue
                
                regressed = stats.linregress(timepoints, data)
                all_regressions.append(regressed[:2])
            
            # Iterate over regressed slopes
            # While the doubling time is decreasing, continue.
            # Stop when it increases, giving the first maximum
            # doubling time. Useful in the event of diauxie, etc.
            all_slopes = np.array([x[0] for x in all_regressions])
            filtered_slopes = all_slopes[np.isfinite(all_slopes)]
            well_dts = (np.log(2) / (filtered_slopes * 60.0))
            best_idx, best_dt = 0, np.inf
            for idx, dt in enumerate(well_dts):
                if dt <= best_dt:
                    best_dt = dt
                    best_idx = idx
                else:
                    break
            doubling_times[well_key] = best_dt
            
        return doubling_times

    def PlotAll(self, label_mapping=None, measurement_interval=30):
        """Plots all the growth curves on the same figure."""
        my_label_mapping = label_mapping or {}
        fig = pylab.figure()
        data = np.exp(self.corrected_log_smoothed_well_df)
        
        for well_key, well_data in data.iteritems():
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
        pylab.ylim(0,5)
    
    def PlotByLabels(self, label_mapping, measurement_interval=30.0):
        """Plots growth curves with the same labels on the same figures."""
        inverse_mapping = label_mapping.InverseMapping()
        smoothed_data = data = np.exp(self.corrected_log_smoothed_well_df)
        
        for descriptive_label, orig_labels in inverse_mapping.iteritems():

            fig = pylab.figure()
            pylab.title(descriptive_label, figure=fig)
            pylab.xlabel('Time (Hours)', figure=fig)
            pylab.ylabel('OD', figure=fig)
            
            for label in orig_labels:
                well_data = smoothed_data[label]
                n_measurements = len(well_data)
                timepoints = np.arange(n_measurements) * measurement_interval / 60.0
                pylab.plot(timepoints, well_data, label=descriptive_label,
                           figure=fig)
                
    def PlotByLabelPrefix(self, label_mapping, measurement_interval=30.0,
                          label_delimiter='+'):
        """Plots growth curves with the same labels on the same figures."""
        extended_mapping = {}
        for orig_label, descriptive_label in label_mapping.iteritems():
            prefix = descriptive_label.split(label_delimiter)[0]
            sub_mapping = extended_mapping.setdefault(prefix, {})
            sub_mapping[orig_label] = descriptive_label
        
        linestyles = ['-', '--', '.-']
        n_styles = len(linestyles)
        smoothed_data = self.smoothed_well_df
        for figure_label, sub_mapping in extended_mapping.iteritems():

            fig = pylab.figure()
            labels = set()
            descriptive_labels = sorted(sub_mapping.values())
            label_to_style = dict((dl, linestyles[i % n_styles])
                                  for i, dl in enumerate(descriptive_labels))
            
            for i, (orig_label, descriptive_label) in enumerate(sub_mapping.iteritems()):
                well_data = smoothed_data[orig_label]
                n_measurements = len(well_data)
                labels.add(descriptive_label)
                linestyle = label_to_style[descriptive_label]
                timepoints = np.arange(n_measurements) * measurement_interval / 60.0
                pylab.plot(timepoints, well_data, label=descriptive_label,
                           linestyle=linestyle, figure=fig)
            
            pylab.title(figure_label, figure=fig)
            pylab.xlabel('Time (Hours)', figure=fig)
            pylab.ylabel('OD', figure=fig)
            pylab.legend()
    
    def PrintByMeanFinalDensity(self, label_mapping):
        inverse_mapping = label_mapping.InverseMapping()
        
        smoothed_data = np.exp(self.corrected_log_smoothed_well_df)
        mean_final_ods = []
        for descriptive_label, orig_labels in inverse_mapping.iteritems():
            sub_data = smoothed_data[orig_labels]
            mean = sub_data.mean(axis=1)
            std = sub_data.std(axis=1) / np.sqrt(len(orig_labels))
            
            mean_final_ods.append((mean[-1], std[-1], descriptive_label))
                
        for final_od, stderr, label in sorted(mean_final_ods, reverse=True):
            print '%s, %0.2g' % (label, final_od)
                                      
    
    def PlotMeanGrowth(self, label_mapping, measurement_interval=30.0,
                       label_delimiter='+'):
        """Plots growth curves with the same labels on the same figures."""
        inverse_mapping = label_mapping.InverseMapping()

        fig = pylab.figure()
        pylab.xlabel('Time (Hours)', figure=fig)
        pylab.ylabel('OD', figure=fig)
        
        linestyles = ['-', '--', '.-']
        n_styles = len(linestyles)
        suffixes = set([dl.split(label_delimiter)[-1]
                        for dl in inverse_mapping.keys()])
        style_mapping = dict((s, linestyles[i%n_styles])
                             for i, s in enumerate(suffixes))
        
        smoothed_data = np.exp(self.corrected_log_smoothed_well_df)
        for descriptive_label, orig_labels in inverse_mapping.iteritems():
            sub_data = smoothed_data[orig_labels]
            mean = sub_data.mean(axis=1)
            stderr = sub_data.std(axis=1) / np.sqrt(len(orig_labels))
            n_measurements = len(mean)
            
            suffix = descriptive_label.split(label_delimiter)[-1]
            linestyle = style_mapping[suffix]
            
            timepoints = np.arange(n_measurements) * measurement_interval / 60.0
            pylab.errorbar(timepoints, mean,
                           yerr=stderr, label=descriptive_label,
                           linestyle=linestyle, linewidth=2)
        
        pylab.legend(loc='upper left', prop={'size':'6'})
                
                


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
    #plate_data.PlotAll()
    #plate_data.PlotDoublingTimeByLabels(well_labels, run_time=23)
    #plate_data.PlotMeanGrowth(well_labels)
    plate_data.PrintByMeanFinalDensity(well_labels)
    pylab.show()
