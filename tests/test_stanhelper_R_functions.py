import os
import numpy as np

from unittest import TestCase
from stanhelper import write_rdump, read_rdump


class TestRfunctions(TestCase):

    datadict = {}
    datadict['J'] = 8
    datadict['y'] = np.array([28, 8, -3, 7, -1, 1, 18, 12])
    datadict['sigma'] = np.array([15, 10, 16, 11, 9, 11, 10, 18])
    datadict['tau'] = 25
    datadict['beta'] = np.diag((4.1, 5.2, 6.345))

    def test_write_rdump(self):
        FILENAME = 'tests/test_write_rdump.data.R'
        if os.path.isfile(FILENAME):
            os.remove(FILENAME)

        write_rdump(self.datadict, FILENAME)
        write_rdump_str = open(FILENAME, 'r').read()

        real_rdump_str = ('J <- 8\n'
                          'y <-\n'
                          'c(28, 8, -3, 7, -1, 1, 18, 12)\n'
                          'sigma <-\n'
                          'c(15, 10, 16, 11, 9, 11, 10, 18)\n'
                          'tau <- 25\n'
                          'beta <-\n'
                          'structure(c(4.1, 0.0, 0.0, 0.0,'
                          ' 5.2, 0.0, 0.0, 0.0, 6.345), .Dim = c(3, 3))\n')

        os.remove(FILENAME)
        self.assertEqual(real_rdump_str, write_rdump_str)

    def test_read_rdump(self):
        FILENAME = 'tests/test_read_rdump.data.R'
        if os.path.isfile(FILENAME):
            os.remove(FILENAME)

        write_rdump(self.datadict, FILENAME)
        read_rdump_dict = read_rdump(FILENAME)

        os.remove(FILENAME)
        np.testing.assert_equal(read_rdump_dict, self.datadict)
