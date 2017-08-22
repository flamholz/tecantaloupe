#!/usr/bin/python

from scipy import stats
from scipy import integrate
import matplotlib.colors as colors
import numpy as np
import pandas
import pylab


class PlateTimeCourse(object):
    """Immutable plate data with convenience methods for computations."""

    def __init__(self, well_df):
        self._well_df = well_df  # Immutable

    def blank(self, n_skip=3, n_av=5):
        """Return a new timecourse that has been blanked.

        """
        # Subtract off the mean of the 5 lowest recorded values
        # for each time series.
        corrected_df = self._well_df.copy()
        
        for key, values in corrected_df.iteritems():
            vals_to_av = corrected_df[key].loc[n_skip:n_skip+n_av]
            corrected_df[key] -= np.mean(vals_to_av)

        return PlateTimeCourse(corrected_df)

    def smooth(self, window=3, rounds=2):
        assert rounds > 0

        smoothed = self._well_df
        for _ in xrange(rounds):
            smoothed = pandas.rolling_mean(smoothed, window)

        return PlateTimeCourse(smoothed)

    def mean_by_name(self, plate_spec):
        """Aggregate cells by PlateSpec name, return means.

        Returns means as a DataFrame.
        """
        mapping = plate_spec.well_to_name_mapping()
        g = self._well_df.groupby(mapping, axis=1)
        return g.mean()

    def sem_by_name(self, plate_spec):
        """Aggregate cells by PlateSpec name, return SEM.

        Returns standard error of the mean as a DataFrame.
        """
        mapping = plate_spec.well_to_name_mapping()
        g = self._well_df.groupby(mapping, axis=1)
        return g.sem()
    
    # methods below are probably broken. keeping code for notes

    def GetDoublingTimes(self, measurement_interval=30,
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

    def PlotMeanAuc(self, label_mapping,
                    label_delimiter='+',
                    include_err=True,
                    prefixes_to_include=None):
        """Plots the mean AUC for each descriptive label in the mapping."""
        aucs = self.GetAreaUnderCurve()
        inverse_mapping = label_mapping.InverseMapping()
        
        fig = pylab.figure()
        pylab.title('Per condition AUC', figure=fig)
        
        mean_aucs = []
        std_errs = []
        labels = []
        for descriptive_label, orig_labels in inverse_mapping.iteritems():
            split_vals = descriptive_label.split(label_delimiter)
            prefix = split_vals[0]
            if (prefixes_to_include is not None and
                prefix not in prefixes_to_include):
                continue
            
            orig_aucs = [aucs[l] for l in orig_labels]
            mean_auc = np.mean(orig_aucs)
            std_err = np.std(orig_aucs) / float(len(orig_labels))
            
            labels.append(descriptive_label)
            mean_aucs.append(mean_auc)
            std_errs.append(std_err)
        
        locs = np.arange(len(labels))
        mean_aucs = np.array(mean_aucs)
        std_errs = np.array(std_errs)
        idxs = np.argsort(mean_aucs)
        
        yerr = std_errs[idxs] if include_err else None
        pylab.bar(locs, mean_aucs[idxs],
                  yerr=std_errs[idxs],
                  fill=False, figure=fig)
        pylab.xticks(locs + 0.5, [labels[i] for i in idxs],
                     rotation=45)

    def PlotAll(self, label_mapping=None, measurement_interval=30):
        """Plots all the growth curves on the same figure."""
        my_label_mapping = label_mapping or {}
        fig = pylab.figure()
        data = self.smoothed_well_df
        #data = np.exp(self.corrected_log_smoothed_well_df)
        
        for well_key, well_data in data.iteritems():
            n_measurements = len(well_data)
            timepoints = np.arange(n_measurements) * measurement_interval
            well_label = my_label_mapping.get(well_key, well_key)
            pylab.plot(timepoints, well_data, label=well_label, figure=fig)
        
        pylab.xlabel('Time (Min)')
        pylab.ylabel('OD')
    
    def PlotDoublingTimeByLabels(self, label_mapping,
                                 measurement_interval=30.0):
        """Produces a boxplot of the per-label doubling times.
        
        Args:
            label_mapping: the mapping from wells to labels.
            run_time: the total running time of the experiment.
            measurement_interval: frequency of measurements, minutes.
        """
        inverse_mapping = label_mapping.InverseMapping()
        doubling_times_and_lags = self.GetDoublingTimes(
            measurement_interval)
        
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
        smoothed_data = self.zeroed_smoothed_well_df
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
        
        smoothed_data = self.zeroed_smoothed_well_df
        mean_final_ods = []
        for descriptive_label, orig_labels in inverse_mapping.iteritems():
            try:
                sub_data = smoothed_data[orig_labels]
                mean = sub_data.mean(axis=1)
                std = sub_data.std(axis=1) / np.sqrt(len(orig_labels))
            
                mean_final_ods.append((mean[-1], std[-1], descriptive_label))
            except:
                continue
                
        for final_od, stderr, label in sorted(mean_final_ods, reverse=True):
            print '%s, %0.2g' % (label, final_od)
    
                     
    def PrintDoublingTimes(self, label_mapping, measurement_interval=30):
    	inverse_mapping = label_mapping.InverseMapping()
        doubling_times = self.GetDoublingTimes(measurement_interval)
        
        mean_dts = []
        for descriptive_label, orig_labels in inverse_mapping.iteritems():
            try:
                dts = [doubling_times[o] for o in orig_labels]
                mean_dt = np.mean(dts)
                std_err_dt = np.std(dts) / np.sqrt(len(orig_labels))
                mean_dts.append((mean_dt, std_err_dt, descriptive_label))
            except:
                continue
        
        for dt, err, label in sorted(mean_dts, reverse=False):
            print '%s, %0.2g +/- %0.2g' % (label, dt, err)
    
    def PlotMeanGrowth(self, label_mapping, measurement_interval=30.0,
                       label_delimiter='+', include_err=False,
                       prefixes_to_include=None):
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
        
        # Skip the first n measurements because they usually suck.
        measurement_offset = 6
        smoothed_data = self.zeroed_smoothed_well_df[measurement_offset:]
        for descriptive_label, orig_labels in inverse_mapping.iteritems():
            try:
                sub_data = smoothed_data[orig_labels]
                mean = sub_data.mean(axis=1)
                stderr = sub_data.std(axis=1) / np.sqrt(len(orig_labels))
                n_measurements = len(mean)
            except:
                continue 
                
            split_vals = descriptive_label.split(label_delimiter)
            prefix = split_vals[0]
            if (prefixes_to_include is not None and
                not prefix in prefixes_to_include):
                continue
            
            linestyle = style_mapping[split_vals[-1]]
            
            timepoints = ((np.arange(n_measurements) +
                           measurement_offset) * measurement_interval / 60.0)
            if include_err:
                pylab.errorbar(timepoints, mean,
                               yerr=stderr, label=descriptive_label,
                               linestyle=linestyle, linewidth=2)
            else:
                pylab.plot(timepoints, mean,
                           label=descriptive_label,
                           linestyle=linestyle, linewidth=2)
        
        pylab.axvspan(72/2, 80/2, facecolor='m', alpha=0.5)
        pylab.legend(loc='upper left', prop={'size':'6'})

