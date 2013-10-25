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
        with open(fname) as f:
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
                cell_label = '%s%02d' % (row_label, i+1)
                cell_data = float(cell)
                well_dict.setdefault(cell_label, []).append(cell_data)
        
        n_measurements = len(well_dict.values()[0])
        timepoints = np.arange(float(n_measurements)) * self._measurement_interval
        data_frame = pandas.DataFrame(well_dict, index=timepoints)

        return PlateTimeCourse(data_frame)
    