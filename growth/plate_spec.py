#!/usr/bin/python

import csv

class PlateSpec(dict):
    
    DEFAULT_COL_LABELS = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
    
    @staticmethod
    def NullMapping():
        return PlateSpec()
    
    @staticmethod
    def FromFile(f):
        """Assumes f is a CSV file."""
        mapping = {}
        for i, row in enumerate(csv.reader(f)):
            assert len(row) == 12, 'Mapping should have 12 columns.'
            assert i <= len(PlateSpec.DEFAULT_COL_LABELS), 'Mapping should have 8 rows.'
            
            row_label = PlateSpec.DEFAULT_COL_LABELS[i]
            for j, new_label in enumerate(row):
                default_label = '%s%d' % (row_label, j+1)
                mapping[default_label] = new_label
        
        return PlateSpec(mapping)

    @staticmethod
    def FromFilename(filename):
        with open(filename, 'U') as f:
            return PlateSpec.FromFile(f)
        
    def InverseMapping(self):
        """Maps new labels to defaults in a list."""
        inverse_mapping = {}
        for orig_label, descriptive_label in self.iteritems():
            inverse_mapping.setdefault(descriptive_label, []).append(orig_label)
        return inverse_mapping
        