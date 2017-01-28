#%%

import numpy
import scipy.stats
import matplotlib.pyplot as plt

sigmas = numpy.array([0.97, 1.05, 1.14, 1.24, 0.95, 0.73, 0.64, 1.58, 0.98, 0.60]) # mean 1, std = 0.25

x = numpy.array([10.83, 8.46, 9.19, 11.00, 10.53, 10.44, 9.34, 7.49, 7.52, 11.25])

w = 1 / (sigmas**2)

print "Single estimate of theta = {0}".format(w.dot(x) / sum(w))

theta = 10.0

ds = []

for r in range(2000):
    x = scipy.stats.norm.rvs(theta, sigmas)

    d = w.dot(x) / sum(w)

    ds.append(d)

plt.hist(ds)
plt.xlabel('$\delta$', fontsize = 16)
plt.ylabel('Count', fontsize = 16)
plt.title('Histogram of 2000 random samples of $\delta$', fontsize = 16)
plt.show()

print "approx. mean delta = {0}".format(numpy.mean(ds))
print "approx. standard deviation delta = {0}".format(numpy.std(ds))
