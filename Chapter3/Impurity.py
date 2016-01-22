# This script displays a chart showing the output of
# various impurity measurements. Impurity measurements
# are used in binary decision trees, likely we won't use
# binary decision trees for our project, but interesting
# nonetheless in my opinion

import matplotlib.pyplot as plt
import numpy as np
 
# Three functions for three types of impurity measurements
# Lots of documentation on these impurity measurements on Wikipedia
def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1 - p))

def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

def error(p):
    return 1 - np.max([p, 1 - p])

# define array x starting at zero
# ending at 1.0, step 0.01
x = np.arange(0.0, 1.0, 0.01)

# Python list comprehension, gotta love it
# look it up on the man pages if you don't know what I'm talking about
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]

# Creates a new figure in plot and a new subplot
fig = plt.figure()

# The subplot definition creates 1 row, 1 column, and selects the first plot
ax = plt.subplot(111)

# This for loop loops through four arrays at once using zip()
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], \
        ['Entropy', 'Entropy (scaled)', \
        'Gini Impurity', 'Misclassification Error'], \
        ['-', '-', '--', '-.'],
        ['black', 'lightgray', 'red', 'green', 'cyan']):
    # creates a new line with given values
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

# This next chunk is all formatting of the plot, look it up
# in the help pages if it's unclear
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),\
        ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()
