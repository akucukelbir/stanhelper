import stanhelper

import numpy as np
import subprocess

# Define data
datadict = {}
datadict['J'] = 8
datadict['y'] = np.array([28,  8, -3,  7, -1,  1, 18, 12])
datadict['sigma'] = np.array([15, 10, 16, 11,  9, 11, 10, 18])
datadict['tau'] = 25

# HELPER FUNCTION 1: save data to disk in Rdump format
stanhelper.stan_rdump(datadict, 'input.data.R')

# Compile cmdStan program; do this however you're most comfortable
# subprocess.call("stanmake eight.stan", shell=True)

# Call cmdStan with whatever parameters you want; 
# eight schools is tough for ADVI so we make it run for lots of iterations
subprocess.call('./eight variational iter=50000 tol_rel_obj=1e-3 \
                data file=input.data.R', shell=True)

# HELPER FUNCTION 2: read results back in dictionary
result = stanhelper.stan_read_csv('output.csv')

# Posterior mean estimates are in a dict `result['mean_pars']`
print 'mu: ',
print result['mean_pars']['mu']
print 'theta: ',
print result['mean_pars']['theta']
print 'tau: ',
print result['mean_pars']['tau']
print

# Samples from the variational posterior are in a dict `result['sampled_pars']`
print 'posterior samples of mu'
print result['sampled_pars']['mu']

