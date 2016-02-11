# Implements a probability mass function, which computes
# the probability mass given an average value for error
# in classification. As can be seen, the ensemble 
# method works well for average classification errors 
# below 0.5, but exacerbates the error in classification
# when the average error is above this value.

from scipy.misc import comb
import numpy as np
import math
import matplotlib.pyplot as plt

def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probs = [comb(n_classifier, k) * \
            error**k * \
            (1 - error)**(n_classifier - k) \
            for k in range(k_start, n_classifier + 1)]
    return sum(probs)

print(ensemble_error(n_classifier=11, error = 0.25))

# Error probabilities from 0 to 1
error_range = np.arange(0.0, 1.01, 0.01)

ens_errors = [ensemble_error(n_classifier=11, error = error) \
        for error in error_range]

plt.plot(error_range, ens_errors, label='Ensemble error', \
        linewidth=2)

plt.plot(error_range, error_range, linestyle='--', \
        label='Base error', linewidth=2)

plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.show()
