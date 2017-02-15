#%%

import numpy
import scipy.stats
import matplotlib.pyplot as plt

N = 200

theta = 2.0
l = 3.0

# Draw N random delivery times
ts = scipy.stats.expon.rvs(size = N, scale = 1 / theta)

# Draw N random numbers of people waiting for the elevotor after each delivery
xs = scipy.stats.poisson.rvs((l * ts), size = N)

#%%

# This is our very high quality Stan model

import pystan

model = """
data {
    int<lower=0> N;
    int x[N];
}

parameters {
    real<lower = 0.0> theta;
    real<lower = 0.0> lambda;
    real<lower = 0.0> t[N];
}

model {
    theta ~ normal(0, 10);

    for (i in 1:N)
    {
        t[i] ~ exponential(theta);
        x[i] ~ poisson(lambda * t[i]);
    }
}

generated quantities {
    vector[N] xhat;

    for (i in 1:N)
    {
        xhat[i] = poisson_rng(lambda * t[i]);
    }
}
"""

m = pystan.StanModel(model_code = model)

#%%

# Run the fit!
fit = m.sampling(data = {
    "N" : N,
    "x" : xs
})

print fit
#%%

# Plot the posterior predictive samples

that = fit.extract()['that']
xhat = fit.extract()['xhat']

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

theta_ = fit.extract()['theta']
lambda_ = fit.extract()['lambda']

#seaborn.distplot(theta_)
#plt.show()

#seaborn.distplot(lambda_)
#plt.show()

df = pandas.DataFrame({ 'theta' : theta_, 'lambda' : lambda_ })

seaborn.pairplot(df)
plt.show()
