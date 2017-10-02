import os
import numpy as np

from unittest import TestCase
from numpy.testing import assert_allclose
from stanhelper.stanhelper import stan_read_csv, get_posterior_estimates, run

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
            'y': np.array([28,  8, -3,  7, -1,  1, 18, 12]),
            'sigma': np.array([15, 10, 16, 11,  9, 11, 10, 18])
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
            'y': np.array([28,  8, -3,  7, -1,  1, 18, 12]),
            'sigma': np.array([15, 10, 16, 11,  9, 11, 10, 18])
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
            'y': np.array([28,  8, -3,  7, -1,  1, 18, 12]),
            # 'sigma': np.array([15, 10, 16, 11,  9, 11, 10, 18])
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
