# Using the wine test data set, there is a section of code to 
# display the difference between normalization and standardization.
# The data is split into 70% training and 30% testing data, at
# which point it is fed into a linear regression model. This model
# prints out the accuracy of the testing and training data to the
# model, both coming to 98% given the conditions. The model also
# prints the amount of zero weights (of 13 feature weights), 
# indicating the sparsity of the weighting by L1 regularization.
# Finally, the script plots the weight value for each feature given
# the value of C fed into the model, noting that the feature
# weights increase significantly for larger C. Recall that C is the
# inverse of the regularization parameter, so larger regularization
# parameter equates to a strongly penalized weight function (underfitting).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Import the wines data set
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

# Assign column names to the pandas DataFrame
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', \
        'Ash', 'Alcalinity of ash', 'Magnesium', \
        'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', \
        'Proanthocyanins', 'Color intensity', 'Hue', \
        'OD280/OD315 of diluted wines', 'Proline']

# Print the unique class labels (what groups do the wines
# belong to?)
print('Class labels', np.unique(df_wine['Class label']))

# Print the first five rows of our data set
print(df_wine.head())

# Split the data set into features and outputs
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values

# Split the feature and output datasets into training and
# testing components, 30% for testing, rest for training
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=0)

mms = MinMaxScaler()

# Fit X_train data to model then transform it, method
# inherited from TransformerMixin class
X_train_norm = mms.fit_transform(X_train)

# Transform X_test data to fitted model
X_test_norm = mms.transform(X_test)

# This is the same as above, except we are standardizing
# rather than normalizing the data
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

# Use L1 regularization to prevent overfitting and encourage sparsity
# IE: Determine inputs with the greatest effect on the outcome and
# minimize the effect of all others on the decision
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training Accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))

# The following block determines how many zero weight values there
# are for each unique output, indicating factors which the L2
# regularization determines do not contribute to the outcome
count = [0, 0, 0]
i = 0
for arr in lr.coef_:
    for data in arr:
        if data == 0:
            count[i] += 1
    i += 1

print(count)

# Plot the regularization path
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', \
        'black', 'pink', 'lightgreen', 'lightblue', \
        'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], \
            label=df_wine.columns[column+1], \
            color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='lower left', \
        #bbox_to_anchor=(1.38, 1.03), \
        ncol=1, fancybox=True)
plt.show()
