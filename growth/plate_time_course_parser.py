#!/usr/bin/python

from growth.plate_time_course import PlateTimeCourse


class PlateTimeCourseParser(object):
    
    def __init__(self):
        pass
    
    def ParseFromFile(self, f):
        """Parse a file-like object.
        
        Args:
            f: the file-like object.
        
        Returns:
            A PlateTimeCourse object.
        """
        raise NotImplementedError()

    def ParseFromFilename(self, fname):
        """Convenience to parse from a filename.
        
        Opens the file, parses.
        
        Args:
            fname: the name/path to the file.
        
        Returns:
            A PlateTimeCourse object.
        """
        with open(fname, 'U') as f:
            return self.ParseFromFile(f)
        

import numpy as np
import pandas


class BremLabTecanParser(PlateTimeCourseParser):
    """Parses data from the BremLab Tecan Infinite reader."""
    
    def __init__(self, measurement_interval=30.0):
        self._measurement_interval = measurement_interval
    
    def ParseFromFile(self, f):
        """Concrete implementation."""
        well_dict = {}
        for line in f:
            if line.startswith('<>'):
                continue  # Skip header lines.
            
            row_data = line.strip().split("\t")
            row_label = row_data[0]
            
            # First entry in line is the row label.
            for i, cell in enumerate(row_data[1:]):
                cell_label = '%s%d' % (row_label, i+1)
                cell = cell.strip() or '0'
                cell_data = float(cell)
                well_dict.setdefault(cell_label, []).append(cell_data)
        
        n_measurements = len(well_dict.values()[0])
        timepoints = np.arange(float(n_measurements)) * self._measurement_interval
        data_frame = pandas.DataFrame(well_dict, index=timepoints)

        return PlateTimeCourse(data_frame)
    

class RineLabSpectramaxParser(PlateTimeCourseParser):
    """Parses data from the Rine Lab Spectramax."""
        
    def __init__(self, measurement_interval=30.0):
        self._measurement_interval = measurement_interval
    
    def _ParseTimestampMinutes(self, tstring):
        split_time = tstring.split(':')
        hours, minutes, seconds = 0, 0, 0
        if len(split_time) == 2:
            minutes, seconds = int(split_time[0]), int(split_time[1])
        elif len(split_time) == 3:
            hours, minutes = int(split_time[0]), int(split_time[1])
            seconds = int(split_time[2])
        else:
            assert False, 'Unrecognized timestamp format %s' % tstring
        
        return float(60 * hours + minutes)
        
    def ParseFromFile(self, f):
        """Concrete implementation."""
        well_dict = {}
        well_names = None
        timestamps = []
        for line in f:
            stripped = line.strip()
            splitted = stripped.split('\t')
            if (len(splitted) < 2 or
                stripped.startswith('#')):
                continue  # Skip comment and empty lines.
            
            if splitted[0].startswith('Plate:'):
                continue # Skip plate metadata
            
            if splitted[2] == 'A1':
                well_names = list(splitted[2:])
                continue
            
            assert well_names, 'Well names not found.'
            
            # Parse the timestamp of the measurement
            measurement_time = splitted[0]
            minutes = self._ParseTimestampMinutes(measurement_time)
            timestamps.append(minutes)
            
            # Parse the per-well measurements at this timestamp
            for well_label, well_measurement in zip(well_names,
                                                    splitted[2:]):
                well_dict.setdefault(well_label, []).append(
                    float(well_measurement))
        
        timestamps = np.array(timestamps)
        data_frame = pandas.DataFrame(well_dict, index=timestamps)
        return PlateTimeCourse(data_frame)
    