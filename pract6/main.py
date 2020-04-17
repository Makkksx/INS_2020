from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from var2 import gen_data
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_data():
    x, y = gen_data(size=1000)
    x, y = shuffle(x, y)
    x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=0.2, random_state=11)
    x_trn = x_trn.reshape(x_trn.shape[0], 50, 50, 1)
    x_tst = x_tst.reshape(x_tst.shape[0], 50, 50, 1)

    encoder = LabelEncoder()
    encoder.fit(y_trn)
    y_trn = encoder.transform(y_trn)
    y_trn = to_categorical(y_trn, 2)

    encoder.fit(y_tst)
    y_tst = encoder.transform(y_tst)
    y_tst = to_categorical(y_tst, 2)
    return x_trn, y_trn, x_tst, y_tst


x_train, y_train, x_test, y_test = get_data()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(50, 50, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='Adagrad',
              metrics=['accuracy'])

h = model.fit(x_train, y_train,
              batch_size=20,
              epochs=10,
              verbose=1,
              validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.title("training and test accuracy")
plt.plot(h.history['acc'], 'g', label='Training acc')
plt.plot(h.history['val_acc'], 'b', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.clf()
plt.title("training and test loss")
plt.plot(h.history['loss'], 'g', label='Training loss')
plt.plot(h.history['val_loss'], 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
