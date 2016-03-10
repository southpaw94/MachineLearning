from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from load_mnist import load_mnist
import numpy as np

X_train, y_train = load_mnist('')
X_test, y_test = load_mnist('', kind='t10k')

y_train_ohe = np_utils.to_categorical(y_train)

np.random.seed(1)

model = Sequential()

model.add(Dense(input_dim=X_train.shape[1],
    output_dim=50, init='uniform',
    activation='tanh'))

model.add(Dense(input_dim=50, output_dim=50,
    init='uniform', activation='tanh'))

model.add(Dense(input_dim=50, output_dim=y_train_ohe.shape[1],
    init='uniform', activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, y_train_ohe, nb_epoch=50,
        batch_size=300, verbose=1, validation_split=0.1,
        show_accuracy=True)

y_train_pred = model.predict_classes(X_train, verbose=0)
print('First 3 predictions: ', y_train_pred[:3])
