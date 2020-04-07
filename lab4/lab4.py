import numpy as np
from numpy import newaxis
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import optimizers
from PIL import Image
import matplotlib.pyplot as plt


def getimage(path):
    return (np.asarray(Image.open(path).convert("L")) / 255.0)[newaxis, :, :]


def createModel(opt, name):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    h = model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0,
                  validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)
    print('test_loss:', test_loss)

    plt.title("{} training and test accuracy".format(name))
    plt.plot(h.history['accuracy'], 'g', label='Training acc')
    plt.plot(h.history['val_accuracy'], 'b', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.clf()
    plt.title("{} training and test loss".format(name))
    plt.plot(h.history['loss'], 'g', label='Training loss')
    plt.plot(h.history['val_loss'], 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()
    return model


mainModel = createModel(optimizers.Adam(), 'Adam')
# createModel(optimizers.sgd(), 'sgd')
# createModel(optimizers.RMSprop(), 'RMSprop')
# createModel(optimizers.Adagrad(), 'Adagrad')
# createModel(optimizers.Nadam(), 'Nadam')

img = getimage('2.bmp')
print(mainModel.predict(img))
print(np.argmax(mainModel.predict(img)))

img = getimage('3.bmp')
print(mainModel.predict(img))
print(np.argmax(mainModel.predict(img)))

img = getimage('4.bmp')
print(mainModel.predict(img))
print(np.argmax(mainModel.predict(img)))

