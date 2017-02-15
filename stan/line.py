#%%

import numpy
import scipy.stats
import matplotlib.pyplot as plt

N = 10

a = 1
b = 2

x = numpy.linspace(0, 1, N)

# Generate our means with some error
y = a * x + b + numpy.random.randn(N) * 0.25

# These are our measurement errors for each y_i
sigmas = numpy.array([0.97, 1.05, 1.14, 1.24, 0.95, 0.73, 0.64, 1.58, 0.98, 0.60]) / 2

#%%

# This is our very high quality Stan model

import pystan

model = """
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
    vector[N] sigmas;
}

parameters {
    real a;
    real b;
}

model {
    for (i in 1:N)
        y[i] ~ normal(a * x[i] + b, sigmas[i]);
}

generated quantities {
    vector[N] yhat;

    for (i in 1:N)
        yhat[i] = normal_rng(a * x[i] + b, sigmas[i]);
}
"""

m = pystan.StanModel(model_code = model)

#%%

# Run the fit!
fit = m.sampling(data = {
    "N" : N,
    "x" : x,
    "y" : y,
    "sigmas" : sigmas
})

print fit
#%%

# Plot the posterior predictive samples

yhat = fit.extract()['yhat']

for i in range(2000):
    plt.plot(x, yhat[-i], 'r*', alpha = 0.05)
plt.plot(x, y)
plt.plot(x, y + sigmas, '--')
plt.plot(x, y - sigmas, '--')
plt.show()

#%%
import seaborn
import pandas

# Plot the parameter posteriors!

a = fit.extract()['a']
b = fit.extract()['b']

seaborn.distplot(a)
plt.show()

seaborn.distplot(b)
plt.show()

df = pandas.DataFrame({ 'a' : a, 'b' : b })

seaborn.pairplot(df)
plt.show()
