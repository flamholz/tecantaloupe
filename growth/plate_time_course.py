#!/usr/bin/python

from scipy import stats
from scipy import integrate
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import pylab


class PlateTimeCourse(object):
    """Immutable plate data with convenience methods for computations."""

    TEMP_COL = 'temp_C'
    TIME_COL = 'time_s'
    SPECIAL_COLS = (TEMP_COL, TIME_COL)

    def __init__(self, well_df):
        self._well_df = well_df  # Immutable

    @property
    def well_df(self):
        """Returns a DataFrame with data of this well.

        TODO: document well_df format.
        """
        return self._well_df

    def labels(self):
        return self._well_df.columns.levels[0].tolist()

    def _filter_columns(self, cols):
        cs = [c for c in cols if c != self.TEMP_COL]
        return cs

    def data_for_plate_wells(self, wells):
        """Grab data only for these wells.

        Returns:
            A new PlateTimeCourse object.
        """
        sorted = self._well_df.sort_index(axis=1)
        
        # keep time column always
        selector = wells + [self.TIME_COL]
        sub_df = self._well_df.loc[:, (slice(None), selector)]
        return PlateTimeCourse(sub_df)

    def data_for_plate_rows(self, rows):
        """Grab only data for these plate rows.

        As opposed to DataFrame rows.

        Returns:
            A new PlateTimeCourse object.
        """
        wells = ['%s%s' % (r, c) for r in rows
                 for c in np.arange(1, 13)]
        # keep time column always
        return self.data_for_plate_wells(wells)

    def data_for_plate_cols(self, cols):
        """Grab only data for these plate columns.

        As opposed to DataFrame columns.

        Returns:
            A new PlateTimeCourse object.
        """
        wells = ['%s%s' % (r, c) for r in 'ABCDEFGH'
                 for c in cols]
        # keep time column always
        return self.data_for_plate_wells(wells)

    def data_for_label(self, label):
        """Returns data for this label.

        Removes the cycle nr. and temp data since
        they get in the way of plotting.
        """
        failure_msg = 'No such label "%s"' % label
        assert label in self._well_df.columns, failure_msg
        data = self._well_df[label]
        data_cols = self._filter_columns(data.columns)
        return data[data_cols]

    def _blank_by_blank_wells(self, blank_wells, n_skip=3, n_av=5):
        """Return a new timecourse that has been blanked.

        TODO: blanking based on separate blank wells should be possible.
        Need to think about that, though, cuz you might have multiple blanks
        with different media which could not be averaged.

        Args:
            n_skip: number of initial points to skip.
            n_av: number of initial points to average on a per-well basis.
        """
        wdf = self._well_df
        index = wdf.index
        pos_to_average = index[n_skip:n_skip+n_av]

        blanked_df = wdf.copy()  # modify a copy

        for dtype in wdf.columns.levels[0]:
            # blank each datatype separately
            sub_df = wdf[dtype]
            cols = set(sub_df.columns)
            cols_to_use = cols.difference(self.SPECIAL_COLS)
            cols_to_use = list(cols_to_use)

            blank_vals = []
            for colname in sub_df.columns:
                if colname not in blank_wells:
                    # don't blank cycle numbers or temperatures.
                    continue

                vals_to_av = sub_df[colname].loc[pos_to_average]
                blank_vals.extend(vals_to_av.values)

            blank_val = np.mean(blank_vals)
            blanked_df.loc[:, (dtype, cols_to_use)] -= blank_val

        return PlateTimeCourse(blanked_df)

    def _blank_by_early_timepoints(self, n_skip=3, n_av=5):
        """Return a new timecourse that has been blanked.

        Args:
            n_skip: number of initial points to skip.
            n_av: number of initial points to average on a per-well basis.
        """
        # Subtract off the mean of the 5 lowest recorded values
        # for each time series.
        corrected_df = self._well_df.copy()
        index = corrected_df.index
        pos_to_average = index[n_skip:n_skip+n_av]

        for key, values in corrected_df.iteritems():
            colname = key[1]
            if colname in (self.TIME_COL, self.TEMP_COL):
                # don't blank cycle numbers or temperatures.
                continue

            vals_to_av = corrected_df[key].loc[pos_to_average]
            corrected_df[key] -= vals_to_av.mean()

        return PlateTimeCourse(corrected_df)

    def blank(self, blank_wells=None, n_skip=3, n_av=5):
        """Return a new timecourse that has been blanked.

        If blank_wells is defined, all wells will be blanked
        according to the mean measurements of n_av early timepoints
        from the defined blanks.

        Otherwise, each well will be blanked separately by subtracting
        off the mean of n_av early timepoints (skipping the first n_skip).

        Args:
            blank_wells: the column IDs of blank wells.
            n_skip: number of initial points to skip.
            n_av: number of initial points to average on a per-well basis.
        """
        if not blank_wells:
            return self._blank_by_early_timepoints(
                n_skip=n_skip, n_av=n_av)
        return self._blank_by_blank_wells(
            blank_wells, n_skip=n_skip, n_av=n_av)

    def blank_by_label(self, label, blanks,
                       max_deviation=None,
                       max_pct_deviation=None):
        """Line up exp and blank at same OD and blank other channels.

        Args:
            label: channel name used to align curves.
            blanks: column names associated with blanks.
            max_deviation: maximum absolute deviation
                allowed for blank and exp points to be associated.
            max_pct_deviation: maximum relative deviation
                allowed for blank and exp points to be associated.

        Returns:
            A new PlateTimeCourse with the averaged data.

        Designed for blanking fluorescence curves against a -fluor control.
        Basic idea:
            1. For each negative control, map mean OD to a timepoint.
            2. For eack exp, make a timecourse of (-) control fluorescence
               at the same OD.
            3. Subtract that signal off to make a blanked signal.

        This code is not optimized for runtime.
        Takes tens of seconds to run on typical data.
        """
        # Grab the measurement we are aligning by
        data_for_alignment = self.data_for_label(label)
        # The other labels are the ones we are blanking
        all_labels = self.labels()
        non_blank_labels = [
            l for l in all_labels if l != label]

        # The other columns are the ones we are blanking.
        cols = data_for_alignment.columns
        non_blank_cols = [
            c for c in cols
            if c not in blanks and not c.startswith(self.TIME_COL)]

        means = data_for_alignment[blanks].mean(axis=1)
        mean_blank_df = pd.DataFrame({
            'time_s': data_for_alignment.time_s,
            'mean_blank': means})

        # Align signals by blank label.
        well2blank_idxs = {}
        for well in non_blank_cols:
            idxs = []
            for tp in data_for_alignment[well]:
                if np.isnan(tp):
                    # Use first index if the value is NaN
                    # Need to do something here.
                    idxs.append(0)
                    continue
                
                abs_diffs = np.abs(mean_blank_df.mean_blank - tp)
                min_diff = np.nanmin(abs_diffs.values)
                # Don't want to use blanks w/ very different values
                if min_diff > (max_deviation or np.inf):
                    idxs.append(np.NaN)
                    continue
                if (100.0*min_diff / tp) > (max_pct_deviation or np.inf):
                    idxs.append(np.NaN)
                    continue

                sorted_idxs = abs_diffs.argsort()
                # First N idxs will be -1 because the values are NaN
                shift = np.sum(sorted_idxs < 0)
                best_match = sorted_idxs[sorted_idxs > 0].iloc[0] + shift + 1
                idxs.append(best_match)
            
            well2blank_idxs[well] = idxs

        # Aggregate blank values for each well
        blank_data = {}
        for l in non_blank_labels:
            ldata = self.data_for_label(l)
            mean_blanks_for_l = ldata[blanks].mean(axis=1)

            for well in non_blank_cols:
                idxs = well2blank_idxs[well]
                blank_data[(l, well)] = mean_blanks_for_l[idxs].values

        blanks_df = pd.DataFrame(blank_data)
        blanked_data_df = self._well_df.copy()
        for l in non_blank_labels:
            blanked = self._well_df[l].copy().subtract(blanks_df[l])
            blanked[self.TIME_COL] = self._well_df[l][self.TIME_COL]
            blanked[self.TEMP_COL] = self._well_df[l][self.TEMP_COL]
            blanked_data_df[l] = blanked

        return PlateTimeCourse(blanked_data_df)

    def smooth(self, window=3, rounds=2):
        """Smooth your data to average out blips.

        TODO: this is also doing a rolling mean on the time
        and temp. Should exclude these from the smoothing.

        Args:
            window: the number of measurements to include
                in the rolling mean window.
            rounds: the number of rounds of smoothing to do.

        Returns:
            A new PlateTimeCourse with the averaged data.
        """
        assert rounds > 0

        smoothed = self._well_df.copy()
        for _ in range(rounds):
            for key, row in smoothed.iteritems():
                colname = key[1]
                if colname in self.SPECIAL_COLS:
                    # don't smooth cycle numbers or temperatures.
                    continue

                smoothed[key] = pd.rolling_mean(row, window)

        return PlateTimeCourse(smoothed)

    def ratio_time_course(self, numerator, denominator):
        """Returns a time course of the ratio of two measurements.

        Args:
            numerator: string label of the numerator of the ratio.
            denominator: string label of the denominator of the ratio.

        Returns:
            A new PlateTimeCourse with a measureming called
                "numerator/denominator."
        """
        num = self.data_for_label(numerator)
        denom = self.data_for_label(denominator)

        num_data = num.drop('time_s', axis=1)
        denom_data = denom.drop('time_s', axis=1)

        name = '%s/%s' % (numerator, denominator)
        ratio_df = num_data / denom_data

        # numerator is the time standard.
        ratio_df[self.TIME_COL] = num[self.TIME_COL]
        full_df = pd.concat(
            [ratio_df], axis=1, keys=[name],
            names=['measurement_type', 'label'])
        return PlateTimeCourse(full_df)

    def mean_by_name(self, plate_spec):
        """Aggregate cells by PlateSpec name, return means.

        Returns means as a DataFrame.
        """
        mapping = plate_spec.well_to_name_mapping()
        means = []

        labels = self.labels()
        for label_name in labels:
            label_df = self._well_df[label_name]
            grouped = label_df.groupby(mapping, axis=1)
            group_means = grouped.mean()
            group_means[self.TIME_COL] = label_df[self.TIME_COL]
            means.append(group_means)

        keys = self._well_df.columns.levels[0]
        merged_df = pd.concat(
            means, axis=1, keys=keys,
            names=['measurement_type', 'label'])

        return PlateTimeCourse(merged_df)

    def std_by_name(self, plate_spec):
        """Aggregate cells by PlateSpec name, return STD DEV.

        Returns standard error of the mean as a DataFrame.
        """
        mapping = plate_spec.well_to_name_mapping()
        stds = []

        labels = self.labels()
        for label_name in labels:
            label_df = self._well_df[label_name]
            grouped = label_df.groupby(mapping, axis=1)
            group_stds = grouped.std()
            group_stds[self.TIME_COL] = label_df[self.TIME_COL]
            stds.append(group_stds)

        keys = self._well_df.columns.levels[0]
        merged_df = pd.concat(
            stds, axis=1, keys=keys,
            names=['measurement_type', 'label'])

        return PlateTimeCourse(merged_df)

    def sem_by_name(self, plate_spec):
        """Aggregate cells by PlateSpec name, return SEM.

        Returns standard error of the mean as a DataFrame.

        This should probably return a PlateTimeCourse object?
        """
        mapping = plate_spec.well_to_name_mapping()
        sems = []

        labels = self.labels()
        for label_name in labels:
            label_df = self._well_df[label_name]
            grouped = label_df.groupby(mapping, axis=1)
            group_sems = grouped.sem()
            group_sems[self.TIME_COL] = label_df[self.TIME_COL]
            sems.append(group_sems)

        keys = self._well_df.columns.levels[0]
        merged_df = pd.concat(
            sems, axis=1, keys=keys,
            names=['measurement_type', 'label'])

        return PlateTimeCourse(merged_df)

    def GrowthYield(self, density_label='OD600'):
        """Computes the maximum density of the culture.

        Calculated as the maximum observed density.
        Recommended that you blank and smooth first.

        Returns:
            A dictionary mapping column names to growth yield.
        """
        OD_data = self.data_for_label(density_label)
        cols = set(OD_data)
        cols_to_use = cols.difference(self.SPECIAL_COLS)

        yields = dict((col, np.nanmax(OD_data[col].values))
                      for col in cols_to_use)
        return yields

    def LagTime(self, density_label='OD600', min_reading=0.1):
        """Returns the lag time in hrs.

        Lag time is here defined as the time at which the culture
        becomes measurable, i.e. crosses the min_reading threshold.

        Returns:
            A dictionary mapping column to lag time in hrs.
        """
        OD_data = self.data_for_label(density_label)
        cols = set(OD_data)
        cols_to_use = cols.difference(self.SPECIAL_COLS)
        cols_to_use = list(cols_to_use)
        time_h = OD_data[self.TIME_COL] / (60.0*60.0)

        # pick timepoint with min abs difference from min_reading
        thresholded = np.abs(OD_data[cols_to_use] - min_reading)
        min_idxs = thresholded.idxmin()

        lags = {}
        for k, idx in min_idxs.iteritems():
            t = time_h[idx]
            lags[k] = t

        max_od = OD_data[cols_to_use].max()
        for k, max_od in max_od.iteritems():
            if max_od < min_reading:
                lags[k] = np.NAN

        return lags

    def GrowthRates(self, density_label='OD600'):
        """Computes the exponential growth rate in gens/hr.

        Returns the local growth rate over time of a 4 measurement window.

        Definitely best to smooth before applying this logic since it
        assumes that derivative(ln(OD)) is smooth.

        TODO: integrate with below? 

        Returns:
            DataFrame of growth rate over time.
        """
        OD_data = self.data_for_label(density_label)
        cols = set(OD_data)
        cols_to_use = cols.difference(self.SPECIAL_COLS)
        time_h = OD_data[self.TIME_COL] / (60.0*60.0)

        growth_rates = {}

        for col in cols_to_use:
            well_data = OD_data[col]
            log_data = np.log(well_data)
            well_slopes = []

            # For each 4-measurement windows.
            # regress against time, keep regression slope.
            for idx in range(len(well_data) - 3):
                local_data = well_data[idx:idx+4].values
                timepoints = time_h[idx:idx+4].values
                regressed = stats.linregress(timepoints, local_data)
                well_slopes.append(regressed[0])

            growth_rates[col] = well_slopes

        growth_rates[self.TIME_COL] = OD_data[self.TIME_COL][:-3]
        return pd.DataFrame(growth_rates)

    def MaxGrowthRates(self, density_label='OD600', min_reading=0.05):
        """Computes the exponential growth rates in gens/hr.

        Maximal growth rate is calculated as the maximal
        exponential growth rate in the growth curve after it crosses
        "min_reading" threshold.

        Definitely best to smooth before applying this logic since it
        assumes that derivative(ln(OD)) is smooth.

        Returns:
            A dictionary mapping column names to growth rates.
        """
        log_lb = np.log(min_reading)
        OD_data = self.data_for_label(density_label)
        cols = set(OD_data)
        cols_to_use = cols.difference(self.SPECIAL_COLS)
        time_h = OD_data[self.TIME_COL] / (60.0*60.0)

        growth_rates = {}

        for col in cols_to_use:
            well_data = OD_data[col]
            log_data = np.log(well_data)
            well_slopes = []

            # For each 4-measurement windows.
            # 1) discard if minimum value beneath user-defined limit.
            # 2) regress against time.
            # 3) keep regression slope.
            for idx in range(len(well_data) - 3):
                local_data = well_data[idx:idx+4].values
                
                if np.nanmin(local_data) < log_lb:
                    well_slopes.append(0.0)
                    continue

                timepoints = time_h[idx:idx+4].values
                regressed = stats.linregress(timepoints, local_data)
                well_slopes.append(regressed[0])

            growth_rates[col] = np.nanmax(well_slopes)

        return growth_rates

