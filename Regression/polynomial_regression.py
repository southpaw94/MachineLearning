import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}

def main():

    df = read_csv('./housing_data.csv')
    X = df[['LSTAT']].values
    y = df['MEDV'].values
    regr = LinearRegression()
    fig, ax = plt.subplots(nrows=1, ncols=2)

    # Create polynomial models
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)

    # Linear fit (d = 1)
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
    regr = regr.fit(X, y)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y, regr.predict(X))
    
    # Quadratic fit (d = 2)
    regr = regr.fit(X_quad, y)
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    quadratic_r2 = r2_score(y, regr.predict(X_quad))

    # Cubic fit (d = 3)
    regr = regr.fit(X_cubic, y)
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    cubic_r2 = r2_score(y, regr.predict(X_cubic))

    # Logarithmic transformation of data
    X_log = np.log(X)
    y_sqrt = np.sqrt(y)
    X_fit_log = np.arange(X_log.min() - 1, X_log.max() + 1, 1)[:, np.newaxis]
    regr = regr.fit(X_log, y_sqrt)
    y_lin_fit_log = regr.predict(X_fit_log)
    linear_r2_log = r2_score(y_sqrt, regr.predict(X_log))
    
    # Plot the data
    ax[0].scatter(X, y,
            label='training points',
            color='lightgray',
            s=40)
    ax[0].plot(X_fit, y_lin_fit, 
            label='linear (d=1), $R^2=%.2f$' % linear_r2,
            color='blue', lw=4, linestyle=':')
    ax[0].plot(X_fit, y_quad_fit,
            label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
            color='red', lw=4, linestyle='-')
    ax[0].plot(X_fit, y_cubic_fit,
            label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
            color='green', lw=4, linestyle='--')

    ax[0].set_xlabel('% lower status of the population [LSTAT]')
    ax[0].set_ylabel('Price in $1000\'s [MEDV]')
    ax[0].legend(loc='upper right')
    ax[0].set_title('Polynomial Regression')

    ax[1].scatter(X_log, y_sqrt,
            label='training points',
            color='lightgray',
            s=40)
    ax[1].plot(X_fit_log, y_lin_fit_log, 
            label='linear (d=1), $R^2=%.2f$' % linear_r2_log,
            color='blue',
            lw=4)
    ax[1].set_xlabel('log(% lower status of the population [LSTAT])')
    ax[1].set_ylabel('$\sqrt{Price \; in \; \$1000\'s [MEDV]}$')
    ax[1].legend(loc='upper right')
    ax[1].set_title('Logarithmic Transformation')

    plt.rc('font', **font)
    plt.show()

if __name__ == '__main__':
    main()
