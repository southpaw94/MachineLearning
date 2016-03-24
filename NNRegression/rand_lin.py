import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.preprocessing import normalize

X = np.arange(1, 101)
y = X * 0.5 + 20
rand = uniform(0, 5, 100)

y += rand

X = X.reshape((100, 1))
y = y.reshape((100, 1))

X_norm = normalize(X, norm='l2')

model = Sequential()
model.add(Dense(input_dim=1, output_dim=128, 
                activation='sigmoid', init='uniform'))
#model.add(Dropout(0.2))
model.add(Dense(input_dim=128, output_dim=1,
                activation='linear', init='uniform'))
sgd = SGD(lr=0.0001, momentum=0.0, decay=0.0)
model.compile(loss='mean_squared_error',
              optimizer='rmsprop')

model.fit(X, y, batch_size=1)

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.show()
