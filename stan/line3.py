#%%
import numpy
import scipy.stats
import matplotlib.pyplot as plt
import seaborn
import pystan

a = 1
b = 2
N = 10

x = numpy.linspace(0, 10, N)
y = a * x + b + numpy.random.randn(N) * 0.5

plt.plot(x, y, '-*')
plt.show()

#%%

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
        y[i] ~ normal(x[i] * a + b, sigma);
}

generated quantities {
    vector[N] yhat;

    for (i in 1:N)
        yhat[i] = normal_rng(x[i] * a + b, sigma);
}
"""

m1 = pystan.StanModel(model_code = model)

#%%

fit = m1.sampling(data = {
    "N" : N,
    "x" : x,
    "y" : y
})

print fit

#%%
a = fit.extract()['a']
b = fit.extract()['b']
sigma = fit.extract()['sigma']

plt.hist(a, bins = 20, alpha = 0.5)
#plt.title('Posterior of a')
#plt.show()

plt.hist(b, bins = 20, alpha = 0.5)
plt.hist(sigma, bins = 20, alpha = 0.5)
plt.legend(['a', 'b', 'sigma'])
#plt.title('Posterior of b')
plt.show()

for i in range(200):
    plt.plot(x, x * a[-i] + b[-i], 'r', alpha = 0.1)
plt.plot(x, y, '-*')
plt.show()

for i in range(200):
    plt.plot(x, x * a[-i] + b[-i] + numpy.random.randn(N) * sigma[-i], 'r', alpha = 0.1)
plt.plot(x, y, '-*')
plt.show()

#%%

a = 1
b = 2
N = 100

x2 = numpy.linspace(0, 10, N)
y2 = a * x2 + b + numpy.random.randn(N) * 0.5

plt.plot(x2, y2, '-*')
plt.show()

#%%

fit = m1.sampling(data = {
    "N" : N,
    "x" : x2,
    "y" : y2
})

print fit

#%%
a = fit.extract()['a']
b = fit.extract()['b']
sigma = fit.extract()['sigma']

plt.hist(a, bins = 20, alpha = 0.5)
#plt.title('Posterior of a')
#plt.show()

plt.hist(b, bins = 20, alpha = 0.5)
plt.hist(sigma, bins = 20, alpha = 0.5)
plt.legend(['a', 'b', 'sigma'])
#plt.title('Posterior of b')
plt.show()

for i in range(200):
    plt.plot(x2, x2 * a[-i] + b[-i], 'r', alpha = 0.1)
plt.plot(x2, y2, '-*')
plt.show()

for i in range(200):
    plt.plot(x2, x2 * a[-i] + b[-i] + numpy.random.randn(N) * sigma[-i], 'r', alpha = 0.1)
plt.plot(x2, y2, '-*')
plt.show()

#%%
