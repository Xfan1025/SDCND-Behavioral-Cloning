'''
Use this script to retrain a model.

I trained a separate model for track2. It drives great for most of the track but off the road at a curve under shadow.

So I collected some new data from the shadow area and used those data to retrain my model. (Transfer Learning) 
'''

import pandas as pd
import numpy as np

import keras
import cv2

from keras.models import load_model


# prepare new data for training
images = []
measurements = []

track2 = pd.read_csv('./../track2_shadow/driving_log.csv')

for path, m in zip(track2.iloc[:, 0], track2.iloc[:, 3]):
    img_path = './../track2_shadow/IMG/' + path.split('\\')[-1]

    img = cv2.imread(img_path)[:,:,::-1] # covert from 'BGR'(cv2 default) to 'RGB'
    images.append(img)
    measurements.append(float(m))

    # flip the image over
    img_flipped = np.fliplr(img)
    images.append(img_flipped)
    measurements.append(-float(m))

assert len(images) == len(measurements)

X_train, y_train = np.array(images), np.array(measurements)

# release memory
del images, measurements, track2

# load pre-trained model
model = load_model('./track2_model.h5')
print(model.summary())
# training
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=3, batch_size=32)
# save the new model
model.save('track2_model_retrained.h5')
