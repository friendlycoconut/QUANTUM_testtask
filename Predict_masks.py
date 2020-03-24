import random

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from skimage.io import imshow
from skimage.transform import resize

from config import *
from utilities import dice_coef, dice_coef_loss, image_processing

X_train, Y_train, X_test = image_processing()


model = load_model('model.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)


preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))


for i in range(10):

    ix = random.randint(0, len(preds_val_t))
    imshow(X_train[int(X_train.shape[0] * 0.9):][ix])
    plt.show()
    imshow(np.squeeze(Y_train[int(Y_train.shape[0] * 0.9):][ix]))
    plt.show()
    imshow(np.squeeze(preds_val_t[ix]))
    plt.show()
