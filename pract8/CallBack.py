from PIL import Image
from keras.callbacks import Callback
import numpy as np


class MyCallBack(Callback):

    def __init__(self, epochs):
        super(MyCallBack, self).__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch not in self.epochs:
            return
        for index, weight in enumerate(self.model.get_weights()):
            if len(weight.shape) == 4:
                _filter = weight.shape[3]
                for j in range(0, weight.shape[2]):
                    for k in range(0, weight.shape[3]):
                        m = weight[:, :, j, k]
                        m *= 255
                        m = np.uint8(m)
                        im = Image.fromarray(m)
                        im_name = f'resources/{index + 1}_{j * _filter + k}_{epoch}.png'
                        im.save(im_name)
