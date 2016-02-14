import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score  

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 
        'Malic acid', 'Ash',
        'Alcalinity of ash',
        'Magnesium', 'Total phenols',
        'Flavanoids', 'Nonflavanoid phenols',
        'Proanthocyanins', 
        'Color intensity', 'Hue',
        'OD280/OD315 of diluted wines',
        'Proline']

df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'Hue']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
tree = DecisionTreeClassifier(criterion='entropy', 
        max_depth = None)
bag = BaggingClassifier(base_estimator=tree, 
        n_estimators=500,
        max_samples = 1.0,
        max_features = 1.0,
        bootstrap = True,
        bootstrap_features = False,
        n_jobs = 1)

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies: %.3f/%.3f'
        % (tree_train, tree_test))

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies: %.3f/%.3f'
        % (bag_train, bag_test))
