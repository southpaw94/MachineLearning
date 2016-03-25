import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split

df = pd.read_csv('BigWheel.csv')
X = df.iloc[:, 1:-2].values
y = df.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
y_train_ohe = np.array(pd.get_dummies(y_train))
y_test_ohe = np.array(pd.get_dummies(y_test))

model = Sequential()

model.add(Dense(input_dim=X_train.shape[1], output_dim=500,
                activation='tanh', init='uniform'))
model.add(Dense(input_dim=500, output_dim=250, 
                activation='tanh', init='uniform'))
model.add(Dense(input_dim=250, output_dim=y_train_ohe.shape[1], 
                activation='softmax', init='uniform'))

sgd = SGD(lr=0.001, momentum=0.9, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, y_train_ohe, batch_size=100, nb_epoch=100, verbose=1,
          validation_split=0.1, show_accuracy=True)

y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)

y_train_pred = np.array(pd.get_dummies(y_train_pred))
y_test_pred = np.array(pd.get_dummies(y_test_pred))

correct_train_predictions = 0
correct_test_predictions = 0

for i in range(X_train.shape[0]):
    if (y_train_pred[i, :] == y_train_ohe[i, :]).all():
        correct_train_predictions += 1

for i in range(X_test.shape[0]):
    if (y_test_pred[i, :] == y_test_ohe[i, :]).all():
        correct_test_predictions += 1

print('Training accuracy: %.2f' % (correct_train_predictions / X_train.shape[0] * 100))
print('Testing accuracy: %.2f' % (correct_test_predictions / X_test.shape[0] * 100))
