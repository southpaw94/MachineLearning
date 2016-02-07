import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.learning_curve import learning_curve

# This program introduces plotting the training and testing accuracy
# overlaid, which gives an idea of how well the training model and
# parameters suit the data. The closer the training and validation
# curves are, the better the model is performing.

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases'+\
        '/breast-cancer-wisconsin/wdbc.data', header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values

# All malignant tumors will be represented as class 1, otherwise, class 0
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, \
        test_size=0.20, random_state=1)

pipe_lr = Pipeline([('scl', StandardScaler()), \
        ('clf', LogisticRegression(penalty='l2', \
        random_state=0))])

# Use the learning_curve function to create a visualization
# of the how well the learning algorithm fits the testing
# and training data.
train_sizes, train_scores, test_scores = \
        learning_curve(estimator=pipe_lr, \
        X = X_train, \
        y = y_train, \
        train_sizes = np.linspace(0.1, 1.0, 10), \
        cv = 10,
        n_jobs = 1)

# Calculate the mean and standard deviations for the train and test
# data.
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis = 1)

# Format the plots, creating shaded regions between the two curves
plt.plot(train_sizes, train_mean, color='blue', \
        marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, \
        train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', \
        marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std,
        test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Model Accuracy with Slight Overfitting')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()
