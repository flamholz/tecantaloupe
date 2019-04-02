#!/usr/bin/python

import unittest
import numpy as np
import pandas
import pylab

from growth.plate_spec import PlateSpec, PlateSpec384


class PlateSpecTest(unittest.TestCase):
    
    def testNullSpec(self):
        null_ps = PlateSpec.NullPlateSpec()
        sample_names = null_ps.df.name

        self.assertEqual(96, sample_names.size)

        for row in sample_names.index:
            for col in sample_names.columns:
                s = '%s%s' % (row, col)
                self.assertEqual(s, sample_names[col][row])

    def testParseSpec(self):
        ps = PlateSpec.FromFile('growth/plate_specs/example_plate_spec.csv')
        names = ps.df.name
        self.assertEqual(names['1']['A'], 'LB BW')
        self.assertEqual(names['6']['D'], 'glyc ls')
        self.assertEqual(names['9']['G'], 'glucon CCMB1 ind')

        # should have loaded induction data.
        induction = ps.df.induction
        self.assertEqual(induction['1']['A'], '0 nM')
        self.assertEqual(induction['6']['D'], '0 nM')
        self.assertEqual(induction['9']['G'], '100 nM')

        well2name = ps.well_to_name_mapping()
        self.assertEqual(well2name['A1'], 'LB BW')

        name2well = ps.name_to_well_mapping()
        self.assertEqual(len(name2well['LB BW']), 4)


class PlateSpec384Test(unittest.TestCase):
    
    def testNullSpec(self):
        null_ps = PlateSpec384.NullPlateSpec()
        sample_names = null_ps.df.name

        self.assertEqual(384, sample_names.size)

        for row in sample_names.index:
            for col in sample_names.columns:
                s = '%s%s' % (row, col)
                self.assertEqual(s, sample_names[col][row])


    def testParseSpec(self):
        ps = PlateSpec384.FromFile('growth/plate_specs/example_plate_spec384.csv')
        names = ps.df.name
        self.assertEqual(names['1']['A'], 'LB BW')
        self.assertEqual(names['6']['D'], 'glyc ls')
        self.assertEqual(names['9']['G'], 'glucon CCMB1 ind')

        well2name = ps.well_to_name_mapping()
        self.assertEqual(well2name['A1'], 'LB BW')
        self.assertEqual(well2name['O24'], 'glucon CCMB1 ind')
        self.assertEqual(well2name['J6'], 'glyc CA')

        name2well = ps.name_to_well_mapping()
        self.assertEqual(len(name2well['LB BW']), 16)
        

if __name__ == '__main__':
    unittest.main()
        