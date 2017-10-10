import os
import numpy as np

from unittest import TestCase
from numpy.testing import assert_allclose
from stanhelper.stanhelper import (stan_read_csv, get_posterior_estimates,
                                   get_posterior_summary, run)

# ===========================================================================
# eight.data.R
# ===========================================================================
# J <- 8
# y <- c(28,  8, -3,  7, -1,  1, 18, 12)
# sigma <- c(15, 10, 16, 11,  9, 11, 10, 18)
# ===========================================================================

# ===========================================================================
# eight.stan
# ===========================================================================
#
# data {
#   int<lower=0> J;          // number of schools
#   real y[J];               // estimated treatment effect (school j)
#   real<lower=0> sigma[J];  // std err of effect estimate (school j)
# }
# parameters {
#   real mu;
#   real theta[J];
#   real<lower=0> tau;
# }
# model {
#   theta ~ normal(mu, tau);
#   y ~ normal(theta, sigma);
# }
# ===========================================================================


class TestMisc(TestCase):

    def test_run_binary_defaults(self):
        FILENAME = 'tests/eight'
        assert os.path.isfile(FILENAME)

        input_data = {
            'J': 8,
            'y': np.array([28, 8, -3, 7, -1, 1, 18, 12]),
            'sigma': np.array([15, 10, 16, 11, 9, 11, 10, 18])
        }

        stan_output = run(FILENAME,
                          input_data,
                          'sample')

        self.assertListEqual(['stdout', 'output'], list(stan_output.keys()))

        result = get_posterior_estimates(stan_output['output'])
        keys = ['lp__', 'accept_stat__', 'stepsize__', 'treedepth__',
                'n_leapfrog__', 'divergent__', 'energy__', 'mu', 'theta',
                'tau']

        self.assertListEqual(keys, list(result.keys()))
        assert_allclose(result['lp__'], np.array([-18.94640541]), rtol=1e0)
        assert_allclose(result['mu'], np.array([8.073827]), rtol=1e0)
        assert_allclose(result['theta'],
                        np.array([12.419558, 7.816183,
                                  5.874417, 7.385253, 4.876437,
                                  5.872688, 11.357495, 8.674288]), rtol=1e0)
        assert_allclose(result['tau'], np.array([7.735637]), rtol=1e0)

    def test_run_binary_init(self):
        FILENAME = 'tests/eight'
        assert os.path.isfile(FILENAME)

        input_data = {
            'J': 8,
            'y': np.array([28, 8, -3, 7, -1, 1, 18, 12]),
            'sigma': np.array([15, 10, 16, 11, 9, 11, 10, 18])
        }

        init_data = {
            'mu': 8,
            'theta': 6 * np.ones(8),
            'tau': 7,
        }

        stan_output = run(FILENAME,
                          input_data,
                          'sample',
                          init_data=init_data)

        self.assertListEqual(['stdout', 'output'], list(stan_output.keys()))

        result = get_posterior_estimates(stan_output['output'])
        keys = ['lp__', 'accept_stat__', 'stepsize__', 'treedepth__',
                'n_leapfrog__', 'divergent__', 'energy__', 'mu', 'theta',
                'tau']

        self.assertListEqual(keys, list(result.keys()))
        assert_allclose(result['lp__'], np.array([-18.94640541]), rtol=1e0)
        assert_allclose(result['mu'], np.array([8.073827]), rtol=1e0)
        assert_allclose(result['theta'],
                        np.array([12.419558, 7.816183,
                                  5.874417, 7.385253, 4.876437,
                                  5.872688, 11.357495, 8.674288]), rtol=1e0)
        assert_allclose(result['tau'], np.array([7.735637]), rtol=1e0)

    def test_run_binary_missing_data(self):
        FILENAME = 'tests/eight'
        assert os.path.isfile(FILENAME)

        input_data = {
            'J': 8,
            'y': np.array([28, 8, -3, 7, -1, 1, 18, 12]),
            # 'sigma': np.array([15, 10, 16, 11, 9, 11, 10, 18])
        }

        stan_output = run(FILENAME,
                          input_data,
                          'sample')

        self.assertListEqual(['error'], list(stan_output.keys()))

    def test_run_not_binary(self):
        FILENAME = 'tests/test_output_sample.csv'
        assert os.path.isfile(FILENAME)

        with self.assertRaises(RuntimeError):
            run(FILENAME, {}, 'optimize')

        with self.assertRaises(RuntimeError):
            run(FILENAME, {}, 'NOT_A_METHOD')

        with self.assertRaises(RuntimeError):
            run('not_a_valid_file', {}, 'optimize')

    def test_get_posterior_summary_sample(self):
        FILENAME = 'tests/test_output_sample.csv'
        assert os.path.isfile(FILENAME)

        result = get_posterior_summary(stan_read_csv(FILENAME))
        keys = ['lp__', 'accept_stat__', 'stepsize__', 'treedepth__',
                'n_leapfrog__', 'divergent__', 'energy__', 'mu', 'theta',
                'tau']
        subkeys = ['mean', 'std', 'five_perc', 'median', 'ninetyfive_perc']

        self.assertListEqual(keys, list(result.keys()))
        for key in keys:
            self.assertListEqual(subkeys, list(result[key].keys()))

        assert_allclose(result['mu']['mean'], np.array([8.073827]))
        assert_allclose(result['mu']['std'], np.array([5.25104899]))
        assert_allclose(result['mu']['five_perc'], np.array([-0.4602165]))
        assert_allclose(result['mu']['median'], np.array([8.11732]))
        assert_allclose(result['mu']['ninetyfive_perc'], np.array([16.19353]))

        assert_allclose(result['theta']['mean'],
                        np.array([12.419558, 7.816183,
                                  5.874417, 7.385253, 4.876437,
                                  5.872688, 11.357495, 8.674288]))
        assert_allclose(result['theta']['std'],
                        np.array([8.61228728, 6.68721892,
                                  7.91427956, 7.39105258, 6.32559623,
                                  7.03889131, 6.97848441, 8.82055328]))
        assert_allclose(result['theta']['five_perc'],
                        np.array([0.1057379, -2.863044,
                                  -8.5084845, -5.1880425, -5.63355,
                                  -6.229635, 0.9704559, -4.523796]))
        assert_allclose(result['theta']['median'],
                        np.array([11.26905, 7.880545,
                                  6.40003, 7.525235, 5.318285,
                                  6.118715, 10.94805, 8.87839]))
        assert_allclose(result['theta']['ninetyfive_perc'],
                        np.array([28.27765, 19.00013,
                                  17.481145, 19.8772, 14.34689,
                                  16.32348, 24.068015, 23.40457]))

    def test_get_posterior_estimates_sample(self):
        FILENAME = 'tests/test_output_sample.csv'
        assert os.path.isfile(FILENAME)

        result = get_posterior_estimates(stan_read_csv(FILENAME))
        keys = ['lp__', 'accept_stat__', 'stepsize__', 'treedepth__',
                'n_leapfrog__', 'divergent__', 'energy__', 'mu', 'theta',
                'tau']

        self.assertListEqual(keys, list(result.keys()))
        assert_allclose(result['lp__'], np.array([-18.94640541]))
        assert_allclose(result['mu'], np.array([8.073827]))
        assert_allclose(result['theta'],
                        np.array([12.419558, 7.816183,
                                  5.874417, 7.385253, 4.876437,
                                  5.872688, 11.357495, 8.674288]))
        assert_allclose(result['tau'], np.array([7.735637]))

    def test_get_posterior_summary_optimize(self):
        FILENAME = 'tests/test_output_optimize.csv'
        assert os.path.isfile(FILENAME)

        result = get_posterior_summary(stan_read_csv(FILENAME))
        keys = ['lp__', 'mu', 'theta', 'tau']

        self.assertListEqual(keys, list(result.keys()))
        for key in keys:
            self.assertIsInstance(result[key], np.ndarray)

        assert_allclose(result['lp__'], np.array([276.575]))
        assert_allclose(result['mu'], np.array([0.765896]))
        assert_allclose(result['theta'],
                        np.array([0.765896, 0.765896,
                                  0.765896, 0.765896, 0.765896,
                                  0.765896, 0.765896, 0.765896]))
        assert_allclose(result['tau'], np.array([4.335870e-16]))

    def test_get_posterior_estimates_optimize(self):
        FILENAME = 'tests/test_output_optimize.csv'
        assert os.path.isfile(FILENAME)

        result = get_posterior_estimates(stan_read_csv(FILENAME))
        keys = ['lp__', 'mu', 'theta', 'tau']

        self.assertListEqual(keys, list(result.keys()))
        assert_allclose(result['lp__'], np.array([276.575]))
        assert_allclose(result['mu'], np.array([0.765896]))
        assert_allclose(result['theta'],
                        np.array([0.765896, 0.765896,
                                  0.765896, 0.765896, 0.765896,
                                  0.765896, 0.765896, 0.765896]))
        assert_allclose(result['tau'], np.array([4.335870e-16]))

    def test_get_posterior_summary_variational(self):
        FILENAME = 'tests/test_output_variational.csv'
        assert os.path.isfile(FILENAME)

        result = get_posterior_summary(stan_read_csv(FILENAME))
        keys = ['mu', 'theta', 'tau']
        subkeys = ['mean', 'std', 'five_perc', 'median', 'ninetyfive_perc']

        self.assertListEqual(keys, list(result.keys()))
        for key in keys:
            self.assertListEqual(subkeys, list(result[key].keys()))

        assert_allclose(result['mu']['mean'], np.array([1.84578]))
        assert_allclose(result['mu']['std'], np.array([3.574338]))
        assert_allclose(result['mu']['five_perc'], np.array([-3.888642]))
        assert_allclose(result['mu']['median'], np.array([1.88189]))
        assert_allclose(result['mu']['ninetyfive_perc'], np.array([7.662146]))

        assert_allclose(result['theta']['mean'],
                        np.array([4.64194, 2.73732,
                                  -0.087907, 2.4676, -0.007914,
                                  1.06119, 5.76268, 1.77541]), rtol=1e-4)
        assert_allclose(result['tau']['mean'], np.array([9.59118]))

    def test_get_posterior_estimates_variational(self):
        FILENAME = 'tests/test_output_variational.csv'
        assert os.path.isfile(FILENAME)

        result = get_posterior_estimates(stan_read_csv(FILENAME))
        keys = ['mu', 'theta', 'tau']

        self.assertListEqual(keys, list(result.keys()))
        assert_allclose(result['mu'], np.array([1.84578]))
        assert_allclose(result['theta'],
                        np.array([4.64194, 2.73732,
                                  -0.087907, 2.4676, -0.007914,
                                  1.06119, 5.76268, 1.77541]), rtol=1e-4)
        assert_allclose(result['tau'], np.array([9.59118]))
