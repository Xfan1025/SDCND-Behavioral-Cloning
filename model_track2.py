
import pandas as pd
import numpy as np

import keras
import cv2

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D


# ## data balancing by removing some of angle of zeros
def load_data(datafile):
    '''
    datafile: path to driving_log.csv

    Return: dataframe with 90% of steering angle less than 0.01 removed
    '''
    # skipped first row since udacity's data contains header but my own data does not
    data = pd.read_csv(datafile)
    # get indices of rows with steering angle less than 0.01
    indices = []
    for i in data.index:
        if -0.008 < data.iloc[i, 3] < 0.01 :
        # if data.iloc[i, 3] == 0:
            indices.append(i)

    # randome generate 80% of the indices
    index_to_remove = sorted(np.random.choice(indices, int(0.7*len(indices)), replace=False))

    # remove the rows with index above
    return data.drop(data.index[index_to_remove], inplace=False)

# read data and split into images and measurements
images = []
measurements = []

track2 = load_data('./../track2/driving_log.csv')

for path, m in zip(track2.iloc[:, 0], track2.iloc[:, 3]):
    img_path = './../track2/IMG/' + path.split('\\')[-1]

    img = cv2.imread(img_path)[:,:,::-1] # covert from 'BGR'(cv2 default) to 'RGB'
    images.append(img)
    measurements.append(float(m))

    # flip the image over to get augmented data
    img_flipped = np.fliplr(img)
    images.append(img_flipped)
    measurements.append(-float(m))

assert len(images) == len(measurements)


X_train, y_train = np.array(images), np.array(measurements)

# release memory
del images, measurements

batch_size = 32
epochs = 2
drop_prob = 0.5


# ### Build the model
# The model is using NVIDIA Architecture

# nvidia
model = Sequential()
### preprocessing layer
# cropping2D layer
model.add(Cropping2D(cropping=((60, 34), (55, 65)), input_shape=(160, 320, 3)))
# normalisation and centraling
model.add(Lambda(lambda x: x / 255.0 - 0.5))

# conv1 depth: 24, kernel=5*5
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
# dropout
#model.add(Dropout(drop_prob))

# conv2 depth: 36, kernel=5*5
model.add(Conv2D(36, 5, 5, subsample=(2,2) , activation='relu'))
# dropout
model.add(Dropout(drop_prob))

# conv3 depth: 48, kernel=3*3
model.add(Conv2D(48, 5, 5, subsample=(2,2) , activation='relu'))
# dropout
#model.add(Dropout(drop_prob))

model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu'))
# dropout
model.add(Dropout(drop_prob))

model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu'))
# dropout
#model.add(Dropout(drop_prob))

# Flatten
model.add(Flatten())

# FC1
model.add(Dense(1164, activation='relu'))
#model.add(Dropout(drop_prob))

# FC2
model.add(Dense(100, activation='relu'))
#model.add(Dropout(drop_prob))

# FC3
model.add(Dense(50, activation='relu'))
#model.add(Dropout(drop_prob))

# FC3
model.add(Dense(10, activation='relu'))
#model.add(Dropout(drop_prob))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
print(model.summary())

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs, batch_size=batch_size)


model.save('model_track2.h5')
print('model saved')
