"""Avi's take on the Brem lab's tecan processing script.

usage: python growthcurves_tecan.py tecanDataFile.asc > outputFileName.txt

normally the script prints to standard out
"""

from scipy import stats

import numpy as np
import sys


def SmoothData(well_dict):
    """
    Args:
        well_dict: a dictionary mapping wells => [timepoint values] as the value
    
    Returns:
        A new well_dict with averages for every point between the two
        points on either side.
    """
    smooth_dict = {}

    for well, values in well_dict.iteritems():
        
        smooth_values = np.zeroes(len(values))
        # Don't average the first or last cuz we can't 
        n = len(values)
        smooth_values[0] = values[0]
        smooth_values[n-1] = values[-1]
        
        for i, value in enumerate(values[1:-1]):
            real_index = i + 1
            previous_and_next = (values[i], values[i+2])
            smooth_values[real_index] = np.mean(previous_and_next)
            
        smooth_dict[well] = map(lambda x: x - min(smooth_values) + 0.00001, smooth_values)
    return smooth_dict

def GetEfficiency(smooth_well_dict):
    
    high_std_dev = []
    std_cutoff = .01
    efficiency_dict = {}

    for well in smooth_well_dict:

        first_two_timepoints = np.array(smooth_well_dict[well][:2])

#        efficiency_dict[well] = np.array(smooth_well_dict[well][-2:]).mean() - first_two_timepoints.mean()
        efficiency_dict[well] = np.sort(np.array(smooth_well_dict[well]))[-2:].mean() - first_two_timepoints.mean()

        if efficiency_dict[well] < 0:
            efficiency_dict[well] = 0

        if first_two_timepoints.std() > std_cutoff:

            high_std_dev.append([well, first_two_timepoints.std()])

    if len(high_std_dev) > 0:
        print "CAUTION these well have a standard deviation of initial 2 values > ", str(std_cutoff)# I guess this matters if the noise from the initial values could throuw off the baseline for efficiency, and even maybe the doubling time. The smoothing should take care of this though(somehow it doesn't).
        for item in high_std_dev:
            print item

    return efficiency_dict

def CorrectCurves(smooth_well_dict):

    corrected_well_dict = {}
    efficiency_dict = {}

    ## transforms the smoothened curve by 1) subtracting the initial value which corresponds to media. 2) log2 transforming and 3)  correcting for growth at high density 

    for well in smooth_well_dict:
        corrected_well_dict[well] = []

        for timepoint in range(len(smooth_well_dict[well])):
            corrected_well_dict[well].append(np.log(smooth_well_dict[well][timepoint]+((smooth_well_dict[well][timepoint]**3)*.8324)))  # the warringer correction. makes no sense. don't fight it.
    return corrected_well_dict

def CalcDoublingTimeAndLag(corrected_well_dict,
                           timepoint_interval=30):
    """
    Args:
        corrected_well_dict:
        timepoint_interval: the number of minutes between readings.
    """
    slope_and_lag_dict = {}
    
    for well in corrected_well_dict:

        all_slopes = []
        for timepoint in range(4, len(corrected_well_dict[well])-4):
            regress_data = stats.linregress(np.arange(timepoint*timepoint_interval, (timepoint+4)*timepoint_interval, timepoint_interval), 
                                            np.array(corrected_well_dict[well][timepoint:(timepoint+4)]))
            #regress_data = stats.linregress([(timepoint)*timepoint_interval, (timepoint+2)*timepoint_interval, (timepoint+4)*timepoint_interval, (timepoint+6)*timepoint_interval, (timepoint+8)*timepoint_interval, (timepoint+10)*timepoint_interval], [corrected_well_dict[well][timepoint], corrected_well_dict[well][timepoint+2], corrected_well_dict[well][timepoint+4], corrected_well_dict[well][timepoint+6], corrected_well_dict[well][timepoint+8], corrected_well_dict[well][timepoint+10]])

            all_slopes.append(regress_data[:2])  ## appends the slope of each linregress

            ## sort and take the biggest slope with the intercept ## 

        all_slopes.sort()
        all_slopes.reverse()
        all_slopes = zip(*all_slopes)
        highest_slope_mean = [np.mean(all_slopes[0][2:8]), np.mean(all_slopes[1][2:8])]
        #highest_slope_mean = [((all_slopes[0][0] + all_slopes[1][0] + all_slopes[2][0]) / 3.0), ((all_slopes[0][1] + all_slopes[1][1] + all_slopes[2][1]) / 3.0)]
        #print well + "\t" + ",".join(map(lambda x: str(x), all_slopes[0]))

        if highest_slope_mean[0] != 0:
            doubling_time = (np.log(2) / highest_slope_mean[0]) / 60

            lag_time = (np.mean(corrected_well_dict[well][0:5]) - highest_slope_mean[1]) / highest_slope_mean[0] / 60
        else:
            doubling_time = -1.0
            lag_time = 24

        if lag_time > 24:
            lag_time = 24

        slope_and_lag_dict[well] = [doubling_time, lag_time]

    return slope_and_lag_dict

def PrintOutput(efficiency_dict, slope_and_lag_dict, corrected_well_dict):

    all_wells = slope_and_lag_dict.keys()
    all_wells.sort()

    print "WELL", "\t", "DOUBLING TIME", "\t", "LAG TIME", "\t", "EFFICIENCY"
    for well in all_wells:
        
        
        print well, "\t", str(slope_and_lag_dict[well][0]), "\t", str(slope_and_lag_dict[well][1]), "\t", str(efficiency_dict[well]), "\t", "\t".join(map(lambda x: str(x), corrected_well_dict[well]))

def ParseTecanOld(input_filename):
    wells = {}
    with open(input_filename) as f:
        for line in f:
            if not line.strip().split():
                # Skip blank lines. changed by chris 5/3/12
                continue
            elif line.startswith('0s') or line[6] == 'C':
                continue
            elif line.startswith('Date'):
                break
            elif len(line[:-1].split()) > 2:
                data = line[:-1].split()
                wells[data[-1]] = [float(OD) for OD in data[:-1]]
            else:
                continue
    
    # Filter out empty wells.
    # NOTE(flamholz): can't use iteritems if deleting inside the loop (I think).
    for well_name, well_values in wells.items():
        if not well_values:
            del wells[well_name]
    return wells

def ParseTecan(input_filename):
    wells = {}
    with open(input_filename) as f:
        for line in f:
            if not line.strip().split():
                # Skip blank lines. changed by chris 5/3/12
                continue
            elif line.startswith('0s') or line[6] == 'C':
                continue
            elif line.startswith('Date'):
                break
            elif len(line[:-1].split()) > 2:
                data = line[:-1].split()
                wells[data[-1]] = [float(OD) for OD in data[:-1]]
            else:
                continue

    # Filter out empty wells.
    # NOTE(flamholz): can't use iteritems if deleting inside the loop (I think).
    for well_name, well_values in wells.items():
        if not well_values:
            del wells[well_name]
    return wells



    

def Main():
    input_filename = sys.argv[1]
    wells = ParseTecan(input_filename)

    smooth_dict = SmoothData(wells)
    efficiency_dict = GetEfficiency(smooth_dict)
    corrected_well_dict = CorrectCurves(smooth_dict)
    slope_and_lag_dict = CalcDoublingTimeAndLag(corrected_well_dict)
    PrintOutput(efficiency_dict, slope_and_lag_dict, smooth_dict)


if __name__ == '__main__':
    Main()