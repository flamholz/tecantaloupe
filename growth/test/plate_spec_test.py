#!/usr/bin/python

import unittest
import numpy as np
import pandas
import pylab

from growth.plate_spec import PlateSpec


class PlateSpecTest(unittest.TestCase):
    
    def testNullSpec(self):
        null_ps = PlateSpec.NullPlateSpec()
        names = null_ps.df.name

        for row in names.index:
            for col in names.columns:
                s = '%s%s' % (row, col)
                self.assertEquals(s, names[col][row])

    def testParseSpec(self):
        ps = PlateSpec.FromFile('growth/plate_specs/example_plate_spec.csv')
        names = ps.df.name
        self.assertEquals(names['1']['A'], 'LB BW')
        self.assertEquals(names['6']['D'], 'glyc ls')
        self.assertEquals(names['9']['G'], 'glucon CCMB1 ind')

        # should have loaded induction data.
        induction = ps.df.induction
        self.assertEquals(induction['1']['A'], '0 nM')
        self.assertEquals(induction['6']['D'], '0 nM')
        self.assertEquals(induction['9']['G'], '100 nM')


if __name__ == '__main__':
    unittest.main()
        