import os
import numpy as np

from unittest import TestCase
from stanhelper import stan_read_csv


class TestCSVfunctions(TestCase):

    def test_no_method(self):
        FILENAME = 'tests/test_output_no_method.csv'
        assert os.path.isfile(FILENAME)

        with self.assertRaises(RuntimeError):
            stan_read_csv(FILENAME)

    def test_sample_invalid(self):
        FILENAME = 'tests/test_output_sample_invalid.csv'
        assert os.path.isfile(FILENAME)

        with self.assertRaises(RuntimeError):
            stan_read_csv(FILENAME)

    def test_optimize_invalid(self):
        FILENAME = 'tests/test_output_optimize_invalid.csv'
        assert os.path.isfile(FILENAME)

        with self.assertRaises(RuntimeError):
            stan_read_csv(FILENAME)

    def test_variational_invalid(self):
        FILENAME = 'tests/test_output_variational_invalid.csv'
        assert os.path.isfile(FILENAME)

        with self.assertRaises(RuntimeError):
            stan_read_csv(FILENAME)

    def test_read_csv_sample(self):
        FILENAME = 'tests/test_output_sample.csv'
        assert os.path.isfile(FILENAME)

        result = stan_read_csv(FILENAME)
        keys = ['lp__', 'accept_stat__', 'stepsize__', 'treedepth__',
                'n_leapfrog__', 'divergent__', 'energy__', 'mu', 'theta',
                'tau']

        self.assertListEqual(keys, list(result.keys()))
        self.assertEqual(result['lp__'].shape, (1000, 1))
        self.assertEqual(result['mu'].shape, (1000, 1))
        self.assertEqual(result['theta'].shape, (1000, 8))
        self.assertEqual(result['tau'].shape, (1000, 1))

    def test_read_csv_optimize(self):
        FILENAME = 'tests/test_output_optimize.csv'
        assert os.path.isfile(FILENAME)

        result = stan_read_csv(FILENAME)
        keys = ['lp__', 'mu', 'theta', 'tau']

        self.assertListEqual(keys, list(result.keys()))
        self.assertEqual(result['lp__'].shape, (1,))
        self.assertEqual(result['mu'].shape, (1,))
        self.assertEqual(result['theta'].shape, (8,))
        self.assertEqual(result['tau'].shape, (1,))

    def test_read_csv_variational(self):
        FILENAME = 'tests/test_output_variational.csv'
        assert os.path.isfile(FILENAME)

        result = stan_read_csv(FILENAME)
        keys = ['mean_pars', 'sampled_pars']
        self.assertListEqual(keys, list(result.keys()))

        keys_param = ['mu', 'theta', 'tau']
        self.assertListEqual(keys_param, list(result['mean_pars'].keys()))
        self.assertListEqual(keys_param, list(result['sampled_pars'].keys()))

        self.assertEqual(result['mean_pars']['mu'].shape, (1,))
        self.assertEqual(result['mean_pars']['theta'].shape, (8,))
        self.assertEqual(result['mean_pars']['tau'].shape, (1,))

        self.assertEqual(result['sampled_pars']['mu'].shape, (1000, 1))
        self.assertEqual(result['sampled_pars']['theta'].shape, (1000, 8))
        self.assertEqual(result['sampled_pars']['tau'].shape, (1000, 1))
