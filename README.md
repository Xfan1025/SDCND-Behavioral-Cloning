Table of Contents(updating)
=================

   * [Behavioral Cloning](#behavioral-cloning)
   * [Check this Video out - less than 2 minutes!](check-this-video-out---2-minutes-watching)
   * [Goal](#goal)
   * [Files & Code Quality](#files-&-code-quality)
      * [Files](#files)
      * [Functional Code](#functional-code)
      * [Code Readability](#code-readability)
   * [Model Architecture](#model-architecture)
      * [Architecture: NVIDIA End-to-End Deep Learning Network](#architecture-nvidia-end-to-end-deep-learning-network)
      * [Objective, Loss function, Learning rate](#objective-loss-function-learning-rate)
      * [Overfitting Handling](#overfitting-handling)
   * [Data Preprocessing Strategy](#data-preprocessing-strategy)
      * [Image Preprocessing](#image-preprocessing)
          * [Cropping](#cropping)
          * [Normalization](#Normalization)
          * [Rotation](#rotation)
          * [Augmentation](#augmentation)
      * [Steering Angle Preprocessing](#steering-angle-preprocessing)
      * [Final Model Architecture](#final-model-architecture)
   * [Training Strategy](#training-strategy)
      * [Transfer Learning](#transfer-learning)
   * [Driving Test Log](#driving-test-log)
   * [Next Challenge](#next)

---

# Behavioral Cloning


# Check this Video Out - 2 minutes watching!
This video shows how my model drives the car in autonomous mode in the precipitous road!

[![YouTube Video](https://img.youtube.com/vi/iT31m75qitY/0.jpg)](https://www.youtube.com/watch?v=iT31m75qitY)

---

# Goal

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model using the driving data collected from simulator
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[train]: ./asset/train_image.png "Training Image"
[flipped]: ./asset/train_image_flipped.png "Flipped Training Image"

[recovery1]: ./asset/recovery1.png "Recovery Image"
[recovery2]: ./asset/recovery2.png "Recovery Image"
[recovery3]: ./asset/recovery3.png "Recovery Image"
[origin]: ./asset/origin_image.png "Original Image"
[flipped]: ./asset/flipped_image.png "Flipped Image"

[bgr]: ./asset/bgr.png "BGR Image"
[rgb]: ./asset/rgb.png "RGB Image"

[input]: ./asset/input.png "Input Image"
[cropped]: ./asset/cropped.png "Cropped Image"

[hist]: ./asset/hist.png "Histogram"
[hist2]: ./asset/hist2.png "Balanced Histogram"

---
# Files & Code Quality

## Files

My project includes the following files:
* `model_track1.py` containing the script to create and train the model
* `model_track2.py` containing the script to create and train a new model for track2
* `drive.py` for driving the car in autonomous mode
* model_track1.h5 containing a trained convolution neural network model
* `model_track2.h5` containing a trained model with the same architecture, but trained on data from track 2
* `README.md` summarizing the development process and results (you reading it!)
* `videos` Please find the links to YouTube videos I recorded at the bottom of this README


(I am switching between different OSs frequently, so the data paths vary a lot.
As a result, I wrote two scripts for generating different models).

(Other utility files such as dependencies will be added soon)


## Functional Code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around track 1 by executing
```sh
python drive.py model_track1.h5
```
and track 2
```sh
python drive.py model_track2.h5
```

## Code readability

The model_track1.py & model_track2.py files contain the code for training and saving the convolution neural network model. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

# Model Architecture and Data collecting Strategy

## Architecture: NVIDIA End-to-End Deep Learning Network

My model is using the [NVIDIA Architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) implemented in Keras.

[![NVIDIA Architecture](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)]


The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras `lambda` layer (code line 143 in model_track1.py).

## Objective, Loss function, Learning rate

The objective is to build a regressor to predict the steering angle, so I use `Mean Square Error` as loss metric.

I use `Adam` as my optimizer since it is a proven good choice for many tasks. I used Keras' default setting for `Adam` optimizer so the learning rate was not tuned manually.

## Overfitting Handling

The model contains dropout layers in order to reduce overfitting (lines 153, 162). I also tried max_pooling. However, it did not significantly improve the performance so I decided not to use pooling.


# Data Preprocessing Strategy

Training data was chosen to keep the vehicle driving on the road. I keep driving in the center of the road as much as possible. And because I only used data from the car's front camera, I recorded data recovering from the left and right sides of the road to the center of the road. That said, I drive to the left and right sides of the road purposely but without recording it down, then I drive back to the center with recording on. This technique should help tell the model that the edges of the road are the areas it should definitely avoid. And it performs great. An alternative solution to this, should be using side cameras for training which is what I am gonna try out next.

## Image Preprocessing

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to avoid those edges as much as possible. These images show what a recovery looks like:

![alt text][recovery1]
![alt text][recovery2]
![alt text][recovery3]

The same process was applied to track two's model as well.

### Cropping

To fit the NVIDIA Architecture, I used Keras `cropping2d` layer to crop the input images.

![text][input]

This image is what my model is actually seeing:

![text][cropped]

### Normalization
To be able to normalize the training data and the test data from simulator while in autonomous mode, I decided to add a Keras `lambda` layer into the structure.

### Rotation
The default color scheme used by `cv2.imread` is `BGR`. However, the image supplied for test in the simulator will be `RGB`. So it makes sense to convert the color scheme from `BGR` back to `RGB`.

Image red by cv2

![alt text][bgr]

Rotated using `image[:,:,::-1]`

![alt text][rgb]

### Augmentation

To augment the data sat, I also flipped images and angles thinking that this would double my data size and also make the model more generalised. For example, here is an image that has then been flipped:

Original image

![alt text][origin]

Flipped image using `np.fliplr(img)`

![alt text][flipped]


## Steering Angle Preprocessing

This analysis of labels (Steering Angles) distribution is quite important.

This is the histogram of steering angle data I collected from track1.

![alt text][hist]

As the data is highly unbalanced, the model may tend to drive straight ahead for most of the time. Because the data is highly unbalanced, any small noise will confuse the model. So to make the model more robust, I decided to try to balance the data. For this data shown above, it's fairly simple, I just removed 95% of steering angle of zeros.

Histogram of the data after 90% of zeros been removed.

![alt text][hist2]

## Final Model Architecture
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 66, 200, 3)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 66, 200, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 14, 47, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       dropout_1[0][0]                  
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 3, 20, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       dropout_2[0][0]                  
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          1342092     flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
====================================================================================================
Total params: 1,595,511
Trainable params: 1,595,511
Non-trainable params: 0
____________________________________________________________________________________________________
```
# Training Strategy

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 192). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3-5 . I used an adam optimizer so that manually tuning the learning rate wasn't necessary.

The final step was to run the simulator to see how well the car was driving around track. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected more data at those spots and used those new data to train my pretrained model. This made my new training really fast and aim to teach the model pay extra attention to those failure points.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.



# Driving Test Log

### Driving Test No.4 (completed second track):

[![text](https://img.youtube.com/vi/iT31m75qitY/0.jpg)](https://www.youtube.com/watch?v=iT31m75qitY)


### Driving Test No.3 (completed first track):

[![text](https://img.youtube.com/vi/ZOaCBuJ7Y2Q/0.jpg)](https://www.youtube.com/watch?v=ZOaCBuJ7Y2Q)

The model is using NVIDIA Architecture. Trained on a relatively large set of (highly unbalanced) data. This test looks great and can be further improved.

Next step will try balance the data as much as possible and also try to use images from side cameras(The above tests are using front camera only).


### Driving Test No.2:

[![text](https://img.youtube.com/vi/9pI-xAmTzIs/0.jpg)](https://www.youtube.com/watch?v=9pI-xAmTzIs)

Modified LeNet Trained on data mostly focusing on curves.



### Driving Test No.1:

[![image1](https://img.youtube.com/vi/SaXprRTi3NU/0.jpg)](https://www.youtube.com/watch?v=SaXprRTi3NU)

Simple ConvNet Trained on a small set of data using a fairly simple model.




# Next Challenge

Build one model trained using data from one track only to be able to complete both two tracks.
