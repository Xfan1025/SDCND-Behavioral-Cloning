import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D


test_data = pd.read_csv('./data/driving_log.csv')

images = []
measurements = []
for path in test_data.iloc[:,0]:
    # img_path is like: 'IMG/center_2016_12_01_13_30_48_287.jpg'
    img_path = './data/' + path
    img = cv2.imread(img_path)

    images.append(img)

    # flip the image
    img_flipped = np.fliplr(img)
    images.append(img_flipped)
    
for m in test_data.iloc[:, 3]:
    measurements.append(float(m))

    # add measurement for flipped image
    measurements.append(-float(m))

X_train, y_train = np.array(images), np.array(measurements)

drop_prob = 0.5

model = Sequential()
# preprocessing layer

# cropping2D layer
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

# normalisation and centraling
model.add(Lambda(lambda x: x / 255.0 - 0.5))

# conv1 depth: 8, kernel=5*5
model.add(Conv2D(6, 5, 5, activation='relu'))
# max_pooling          
model.add(MaxPooling2D((3, 3)))
# dropout
model.add(Dropout(drop_prob))

# conv2 depth: 16, kernel=3*3
model.add(Conv2D(16, 3, 3, activation='relu'))
# max_pooling          
model.add(MaxPooling2D((3, 3)))
# dropout
model.add(Dropout(drop_prob))

# Flatten
model.add(Flatten())

# FC1
model.add(Dense(1025, activation='relu'))
model.add(Dropout(drop_prob))

# FC2
model.add(Dense(512, activation='relu'))
model.add(Dropout(drop_prob))

# FC3
model.add(Dense(128, activation='relu'))
model.add(Dropout(drop_prob))

# FC3
model.add(Dense(32, activation='relu'))
model.add(Dropout(drop_prob))

model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
print('model saved')