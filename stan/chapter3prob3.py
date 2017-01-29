#%%

import numpy
import scipy.stats
import matplotlib.pyplot as plt

sigmas = numpy.array([0.97, 1.05, 1.14, 1.24, 0.95, 0.73, 0.64, 1.58, 0.98, 0.60]) # mean 1, std = 0.25

theta = 10.0

w = 1 / (sigmas**2)

#%%

import pystan

model = """
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] sigmas;
}

parameters {
    real theta;
}

model {
    for (i in 1:N)
        x[i] ~ normal(theta, sigmas[i]);
}
"""

m = pystan.StanModel(model_code = model)

#%%

#x = scipy.stats.norm.rvs(theta, sigmas)
x = numpy.array([10.83, 8.46, 9.19, 11.00, 10.53, 10.44, 9.34, 7.49, 7.52, 11.25])

fit = m.sampling(data = {
    "N" : len(x),
    "x" : x,
    "sigmas" : sigmas
})

thetas = fit.extract()['theta']

plt.hist(thetas)
plt.title('2000 samples of posterior (from Stan) of $\\theta$ given $x$'.format(w.dot(x) / sum(w)), fontsize = 16)
plt.xlabel('$\\theta$', fontsize = 16)
plt.ylabel('Count', fontsize = 16)
plt.show()

print "approx. mean theta = {0}".format(numpy.mean(thetas))
print "approx. standard deviation theta = {0}".format(numpy.std(thetas))

print 1/sum(w)
print numpy.sqrt(1/sum(w))