import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter = 20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2
            self.cost_.append(cost)
        
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue', s=40)
    plt.plot(X, model.predict(X), color='red', lw=4)
    return None

def main():

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
            header = None,
            sep = '\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
            'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
            'LSTAT', 'MEDV']
    print(df.head())

    # Select a subset of the features and plot the correlation between features
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    sns.pairplot(df[cols], size=2.5);
    plt.title('Correlations between 5 features')
    plt.show()

    # Plot a heatmap of the same subset of features
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=2.5)
    hm = sns.heatmap(cm,
            cbar = True,
            annot = True,
            square = True,
            fmt = '.2f',
            annot_kws = {'size': 15},
            yticklabels = cols,
            xticklabels = cols)
    plt.show()

    X = df[['RM']].values
    y = df['MEDV'].values

    sc_x = StandardScaler()
    sc_y = StandardScaler()

    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y)
    
    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)

    plt.plot(range(1, lr.n_iter + 1), lr.cost_)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.show()

    lin_regplot(X_std, y_std, lr)
    plt.xlabel('Average number of rooms [RM] (standardized)')
    plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
    plt.show()
    
    # Example classification for a house with 5 rooms
    num_rooms_std = sc_x.transform([5.0])
    price_std = lr.predict(num_rooms_std)
    print("Price in $1000's: %.3f" % \
            sc_y.inverse_transform(price_std))
if __name__ == '__main__':
    main()
