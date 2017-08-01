import numpy as np
import subprocess

from stanhelper import write_rdump, stan_read_csv, get_posterior_estimates


# Define data
datadict = {}
datadict['J'] = 8
datadict['y'] = np.array([28, 8, -3, 7, -1, 1, 18, 12])
datadict['sigma'] = np.array([15, 10, 16, 11, 9, 11, 10, 18])
datadict['tau'] = 25

# HELPER FUNCTION 1: save data to disk in Rdump format
write_rdump(datadict, 'input.data.R')

# # Compile cmdStan program; do this however you're most comfortable
# stanmake refers to a helper function for compiling stan models:
# https://gist.github.com/altosaar/7690801f4280ee04a1e4b92c85c40e7e
# subprocess.call("stanmake eight.stan", shell=True)

# Call cmdStan with whatever parameters you want;

# subprocess.call('./eight variational \
#                 data file=input.data.R', shell=True)

# subprocess.call('./eight sample \
#                 data file=input.data.R', shell=True)

# subprocess.call('./eight optimize \
#                 data file=input.data.R', shell=True)


# HELPER FUNCTION 2: read results back in dictionary
result = stan_read_csv('output.csv')

mean_pars = get_posterior_estimates(result)

# Posterior mean estimates are in a dict `result['mean_pars']`
print('mu: ',)
print(mean_pars['mu'])
print('theta: ',)
print(mean_pars['theta'])
print('tau: ',)
print(mean_pars['tau'])

print(mean_pars)

# # Samples from the variational posterior are in a dict `result['sampled_pars']`
# print('posterior samples of mu')
# print(result['sampled_pars']['mu'])
