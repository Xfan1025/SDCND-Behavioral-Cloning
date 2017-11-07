Table of Contents(updateing)
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
      * [Image Preprocessing](#data-preprocessing)
      * [Steering Angle Preprocessing](#steering-angle-preprocessing)
   * [Training Strategy](#training-strategy)
      * [Building an Overfitted Model with Minimal Data](#building-an-overfitted-model-with-minimal-data)
      * [Building a Regularized Model with Augmented Data](#building-a-regularized-model-with-augmented-data)
   * [Next](#next)

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
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"



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

(Using model_track1.py for reference)

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 143).

## Objective, Loss function, Learning rate

The objective is to build a regressor to predict the steering angle, so I use `Mean Square Error` as loss metric.

I use `Adam` as my optimizer since it is a proven good choice for many tasks. I used Keras' default setting for `Adam` optimizer so the learning rate was not tuned manually.

## Overfitting Handling

The model contains dropout layers in order to reduce overfitting (lines 153, 162). I also considered max_pooling. However, it did not significantly improve the performance so I decided not to use pooling.



# Data Preprocessing Strategy

Training data was chosen to keep the vehicle driving on the road. I keep driving in the center of the road as much as possible. And because I only used data from the car's front camera, I used a data recovering from the left and right sides of the road to the center of the road. That said, I drive to the left and right sides of the road purposely but without recording it down, then I drive back to the center with recording on. This technique should help tell the model the edges of the road are the areas it should definitely avoid. And it performs great. An alternative solution to this, should be using side cameras for training which is what I am gonna try out next.

## Image Preprocessing

## Steering Angle Preprocessing

# Training Strategy

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 192). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 1. Solution Design Approach



The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data as stated above.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3-5 . I used an adam optimizer so that manually tuning the learning rate wasn't necessary.
