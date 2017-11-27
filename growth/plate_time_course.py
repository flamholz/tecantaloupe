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

    def blank(self, n_skip=3, n_av=5):
        """Return a new timecourse that has been blanked.

        TODO: blanking based on separate blank wells should be possible.
        Need to think about that, though, cuz you might have multiple blanks
        with different media which could not be averaged.

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
        for _ in xrange(rounds):
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

        This should probably return a PlateTimeCourse object?
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

    def GrowthRates(self, density_label='OD600'):
        """Computes the exponential growth rate in gens/hr.

        Returns the local growth rate over time of a 4 measurement window.

        Definitely best to smooth before applying this logic since it
        assumes that derivative(ln(OD)) is smooth.

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
            for idx in xrange(len(well_data) - 3):
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
            for idx in xrange(len(well_data) - 3):
                local_data = well_data[idx:idx+4].values
                
                if np.nanmin(local_data) < log_lb:
                    well_slopes.append(0.0)
                    continue

                timepoints = time_h[idx:idx+4].values
                regressed = stats.linregress(timepoints, local_data)
                well_slopes.append(regressed[0])

            growth_rates[col] = np.nanmax(well_slopes)

        return growth_rates

