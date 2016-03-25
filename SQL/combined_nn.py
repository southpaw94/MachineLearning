import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split

df = pd.read_csv('BothDrives.csv')
X = df.iloc[:, 1:-3].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

y_train_ohe = np_utils.to_categorical(y_train)

model = Sequential()
model.add(Dense(input_dim=X.shape[1], output_dim=500,
                activation='relu', init='uniform'))
model.add(Dense(input_dim=2000, output_dim=1500,
                activation='relu', init='uniform'))
model.add(Dense(input_dim=1500, output_dim=1000,
                activation='tanh', init='uniform'))
model.add(Dense(input_dim=1000, output_dim=500,
                activation='tanh', init='uniform'))
model.add(Dense(input_dim=500, output_dim=250,
                activation='tanh', init='uniform'))
model.add(Dense(input_dim=250, output_dim=y_train_ohe.shape[1],
                activation='softmax', init='uniform'))

sgd = SGD(lr=0.001, momentum=0.9, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, y_train_ohe, batch_size=300, nb_epoch=100,
          verbose=1, show_accuracy=True, validation_split=0.1)

y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)

print('Training accuracy: %.2f' % (len(y_train[y_train == y_train_pred]) / len(y_train) * 100))
print('Testing accuracy: %.2f' % (len(y_test[y_test == y_test_pred]) / len(y_test) * 100))
