import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
# from MajorityVoteClassifier import MajorityVoteClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.ensemble import VotingClassifier
from itertools import product
from sklearn.grid_search import GridSearchCV

# This program is similar to iris.py, but instead of using
# our own majority voting classifier algorithm, we make use
# of the one already provided by scikit-learn.

# Start by loading the iris dataset from sklearn.datasets
iris = datasets.load_iris()
X, y = iris.data[50:, [1,2]], iris.target[50:]

# Create our label encoder and encode the targets in the range
# [0,1]
le = LabelEncoder()
y = le.fit_transform(y)

# Split into training/testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=0.5, random_state=1)

# Set up our three classifiers used in the majority voting
clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
clf2 = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, metric='minkowski', 
        p=2)

# Create two pipelines to scale the two classifiers which
# require the data to be scaled
pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

# Print the 'area-under-curve' scores for the training data
# under each individual classifier
clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
print('10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
            X = X_train,
            y = y_train,
            cv = 10,
            scoring = 'roc_auc')

    print("ROC AUC: %0.2f (+/- %0.2f) [%s]" %
            (scores.mean(), scores.std(), label))

# Create the majority voting classifier
mv_clf = VotingClassifier(
        estimators = [('p1', pipe1), 
            ('clf2', clf2), 
            ('p3', pipe3)],
        voting = 'soft')

# Add a majority voting label to our set of labels
clf_labels += ['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]

# Print the test data scores for each individual
# classifier, and the majority vote
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator = clf,
            X = X_train, 
            y = y_train,
            cv = 10,
            scoring = 'roc_auc')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" %
            (scores.mean(), scores.std(), label))

# Plot the receiver operating characteristic curves
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls in \
        zip(all_clf, clf_labels, colors, linestyles):
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true = y_test, \
            y_score = y_pred)
    roc_auc = auc(x = fpr, y = tpr)
    plt.plot(fpr, tpr,
            color=clr, 
            linestyle=ls,
            label='%s (auc = %0.2f)' % (label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],
        linestyle='--',
        color='gray',
        linewidth=2)

plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positives Rate')
plt.ylabel('True Positive Rate')
plt.show()

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), 
        np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2, 
        sharex='col', 
        sharey='row',
        figsize=(7, 5))

for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf,
        clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alhpa=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
            X_train_std[y_train==0, 1],
            c='blue',
            marker='^',
            s=50)

    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0],
            X_train_std[y_train==1, 1],
            c='red',
            marker='o',
            s=50)

    axarr[idx[0], idx[1]].set_title(tt)

plt.text(-4, -4.5, 
        s='Sepal width [standardized]', 
        ha = 'center', va='center', fontsize=12)
plt.text(-11.5, 4.5, 
        s='Petal length [standardized]',
        ha = 'center', va = 'center',
        fontsize=12, rotation=90)
plt.show()

params = {'clf2__max_depth': [1, 2], 'p1__clf__C': [0.001, 0.1, 100]}
grid = GridSearchCV(estimator=mv_clf, param_grid = params,
        cv = 10, scoring = 'roc_auc')
grid.fit(X_train, y_train)

for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f+/-%0.2f %r"
            % (mean_score, scores.std() / 2, params))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)
