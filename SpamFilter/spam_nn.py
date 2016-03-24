from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

def load_data():
    spam_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', header=None)
    X = np.array(spam_data.iloc[:, :-1])
    y = spam_data.iloc[:, -1]
    return X, y

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

y_train_ohe = np_utils.to_categorical(y_train)

model = Sequential()

model.add(Dense(input_dim=X_train.shape[1], output_dim=2500, activation='tanh', init='uniform'))

for i in range(2400, 0, -300):
    model.add(Dense(input_dim=i + 100, output_dim=i, init='uniform', activation='tanh'))
    model.add(Dropout(0.25))

model.add(Dense(input_dim=300, output_dim=y_train_ohe.shape[1], activation='softmax', init='uniform'))

sgd = SGD(lr=0.001, momentum=0.5, decay=1e-7)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, y_train_ohe, nb_epoch=100, 
          batch_size=300, verbose=1, validation_split=0.1,
          show_accuracy=True)

y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)

print('Test accuracy: %.2f' % (len(y_test[y_test==y_test_pred]) / len(y_test) * 100))
print('Train accuracy: %.2f' % (len(y_train[y_train==y_train_pred]) / len(y_train) * 100))

