#%%

import numpy
import scipy.stats
import matplotlib.pyplot as plt

N = 10

a = 1
b = 2

x = numpy.linspace(0, 10, N)

# Generate our data with some error
#y = a * x + b + numpy.random.randn(N) * 0.5

y = numpy.array([2.46, 4.37, 4.55, 5.72, 6.04, 6.82, 8.91, 9.53, 10.0, 12.2])

#%%

# This is our very high quality Stan model

import pystan

model = """
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}

parameters {
    real a;
    real b;
    real sigma;
}

model {
    for (i in 1:N)
        y[i] ~ normal(a * x[i] + b, sigma);
}

generated quantities {
    vector[N] yhat;

    for (i in 1:N)
        yhat[i] = normal_rng(a * x[i] + b, sigma);
}
"""

m = pystan.StanModel(model_code = model)

#%%

# Run the fit!
fit = m.sampling(data = {
    "N" : N,
    "x" : x,
    "y" : y
})

print fit
#%%

# Plot the posterior predictive samples

yhat = fit.extract()['yhat']

for i in range(200):
    plt.plot(x, yhat[-i], 'r', alpha = 0.1)
plt.plot(x, y, '-*')
plt.show()

#%%
import seaborn
import pandas

# Plot the parameter posteriors!

a = fit.extract()['a']
b = fit.extract()['b']
sigma = fit.extract()['sigma']

plt.hist(a, 20, alpha = 0.5)
plt.hist(b, 20, alpha = 0.5)
plt.hist(sigma, 20, alpha = 0.5)
plt.legend(['a', 'b', 'sigma'])
plt.show()

df = pandas.DataFrame({ 'a' : a, 'b' : b, 'sigma' : sigma })

seaborn.pairplot(df)
plt.show()
