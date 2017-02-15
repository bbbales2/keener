#%%

import numpy
import scipy.stats
import matplotlib.pyplot as plt
import seaborn
import pandas
#%%
# Generating data
N = 10
y = [0, 1]
for n in range(N - 2):
    y.append(y[-1] + y[-2])

# Adding a little noise
x = numpy.arange(0, N)
y = numpy.array(y) + numpy.random.randn(N) * 0.5

plt.plot(x, y, '-*')
plt.title('Data')
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
plt.title('Linear fit')
plt.show()

for i in range(200):
    plt.plot(x, x * a[-i] + b[-i] + numpy.random.randn(N) * sigma[-i], 'r', alpha = 0.1)
plt.plot(x, y, '-*')
plt.title('Linear fit w/ estimated noise')
plt.show()
#%%
a = fit.extract()['a']
b = fit.extract()['b']
sigma = fit.extract()['sigma']

df = pandas.DataFrame({ 'a' : a, 'b' : b, 'sigma' : sigma })

seaborn.pairplot(df)
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
    real sigma;
}

model {
    for (i in 3:N)
        y[i] ~ normal(a * y[i - 1] + a * y[i - 2], sigma);
}

generated quantities {
    vector[N] yhat;

    yhat[1] = y[1];
    yhat[2] = y[2];

    for (i in 3:N)
        yhat[i] = normal_rng(a * y[i - 1] + a * y[i - 2], sigma);
}
"""

m2 = pystan.StanModel(model_code = model)

#%%

fit = m2.sampling(data = {
    "N" : N,
    "x" : x,
    "y" : y
})

print fit
#%%
yhat = fit.extract()['yhat']

for i in range(200):
    plt.plot(x[2:], yhat[-i, 2:], 'r', alpha = 0.05)
plt.plot(x[2:], y[2:])
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
    for (i in 3:N)
        y[i] ~ normal(a * y[i - 1] + b * y[i - 2], sigma);
}

generated quantities {
    vector[N] yhat;

    yhat[1] = y[1];
    yhat[2] = y[2];

    for (i in 3:N)
        yhat[i] = normal_rng(a * y[i - 1] + b * y[i - 2], sigma);
}
"""

m3 = pystan.StanModel(model_code = model)

#%%

fit = m3.sampling(data = {
    "N" : N,
    "x" : x,
    "y" : y
})

print fit
#%%
yhat = fit.extract()['yhat']

for i in range(200):
    plt.plot(x[2:], yhat[-i, 2:], 'r', alpha = 0.1)
plt.plot(x[2:], y[2:])
plt.show()
#%%
a = fit.extract()['a']
b = fit.extract()['b']
sigma = fit.extract()['sigma']

df = pandas.DataFrame({ 'a' : a, 'b' : b, 'sigma' : sigma })

seaborn.pairplot(df)
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
    a - b ~ normal(0, 0.25);

    for (i in 3:N)
        y[i] ~ normal(a * y[i - 1] + b * y[i - 2], sigma);
}

generated quantities {
    vector[N] yhat;

    yhat[1] = y[1];
    yhat[2] = y[2];

    for (i in 3:N)
        yhat[i] = normal_rng(a * y[i - 1] + b * y[i - 2], sigma);
}
"""

m4 = pystan.StanModel(model_code = model)

#%%

fit = m4.sampling(data = {
    "N" : N,
    "x" : x,
    "y" : y
})

print fit
#%%
yhat = fit.extract()['yhat']

for i in range(200):
    plt.plot(x[2:], yhat[-i, 2:], 'r', alpha = 0.1)
plt.plot(x[2:], y[2:])
plt.show()
#%%
a = fit.extract()['a']
b = fit.extract()['b']
sigma = fit.extract()['sigma']

df = pandas.DataFrame({ 'a' : a, 'b' : b, 'sigma' : sigma })

seaborn.pairplot(df)
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
    real sigma;
}

model {
    for (i in 3:N)
        y[i] ~ normal(a * y[i - 1] + a * y[i - 2], sigma);
}

generated quantities {
    vector[N] yhat;

    yhat[1] = y[1];
    yhat[2] = y[2];

    for (i in 3:N)
        yhat[i] = normal_rng(a * yhat[i - 1] + a * yhat[i - 2], sigma);
}
"""

m5 = pystan.StanModel(model_code = model)

#%%

fit = m5.sampling(data = {
    "N" : N,
    "x" : x,
    "y" : y
})

print fit
#%%
yhat = fit.extract()['yhat']

for i in range(200):
    plt.plot(x[2:], yhat[-i, 2:], 'r', alpha = 0.1)
plt.plot(x[2:], y[2:])
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
        if (y[i] > 0.0)
            log(y[i]) ~ normal(a * x[i] + b, sigma);
}

generated quantities {
    vector[N] yhat;

    for (i in 1:N)
        yhat[i] = exp(normal_rng(a * x[i] + b, sigma));
}
"""

m6 = pystan.StanModel(model_code = model)

#%%

fit = m6.sampling(data = {
    "N" : N,
    "x" : x,
    "y" : y
})

print fit
#%%
yhat = fit.extract()['yhat']

for i in range(200):
    plt.plot(x[2:], yhat[-i, 2:], 'r', alpha = 0.1)
plt.plot(x[2:], y[2:])
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
    a ~ normal(0, 1);
    b ~ normal(0, 1);

    for (i in 1:N)
        y[i] ~ normal(exp(a * x[i] + b), sigma);
}

generated quantities {
    vector[N] yhat;

    for (i in 1:N)
        yhat[i] = normal_rng(exp(a * x[i] + b), sigma);
}
"""

m61 = pystan.StanModel(model_code = model)

#%%

fit = m61.sampling(data = {
    "N" : N,
    "x" : x,
    "y" : y
})

print fit
#%%
yhat = fit.extract()['yhat']

for i in range(200):
    plt.plot(x[2:], yhat[-i, 2:], 'r', alpha = 0.1)
plt.plot(x[2:], y[2:])
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
    real sigma;
}

model {
    sigma ~ normal(0, 10);

    for (i in 1:N)
        y[i] ~ normal((pow(a, x[i]) - pow(-a, -x[i])) / (2 * a - 1), sigma);
}

// We can generated samples directly in Stan
generated quantities {
    vector[N] yhat;

    for (i in 1:N)
        yhat[i] = normal_rng((pow(a, x[i]) - pow(-a, -x[i])) / (2 * a - 1), sigma);
}
"""

m7 = pystan.StanModel(model_code = model)

#%%

fit = m7.sampling(data = {
    "N" : N,
    "x" : x,
    "y" : y
})

print fit
#%%
yhat = fit.extract()['yhat']

for i in range(200):
    plt.plot(x, yhat[-i], 'r', alpha = 0.1)
plt.plot(x, y)
plt.title('Linear fit w/ noise')
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
    real<lower = 0.5> a;
    real sigma;
}

model {
    sigma ~ normal(0, 10);

    for (i in 1:N)
        y[i] ~ normal((pow(a, x[i]) - pow(-a, -x[i])) / (2 * a - 1), sigma);
}

generated quantities {
    vector[N] yhat;

    for (i in 1:N)
        yhat[i] = normal_rng((pow(a, x[i]) - pow(-a, -x[i])) / (2 * a - 1), sigma);
}
"""

m72 = pystan.StanModel(model_code = model)

#%%

fit = m72.sampling(data = {
    "N" : N,
    "x" : x,
    "y" : y
})

print fit
#%%
yhat = fit.extract()['yhat']

for i in range(200):
    plt.plot(x[2:], yhat[-i, :2], 'r', alpha = 0.1)
plt.plot(x[2:], y[2:])
plt.show()
