# Image_Processing
This repository contains all the programming work done by Ian Feekes (ianfeekes@gmail.com) for the University of San Diego MS. Applied Artificial Intelligence and Machine Learning program.

## Assignment 1
The first assignment was two part. 
* The first involved using the MNIST fashion clothing dataset and performing multi-classification for each image. This is done through the neural networks' output layer of 10 output neurons, each representing a class that an image most closely represents, activated through softmax activation since this is a problem of categorical classification. It has an input layer of images converted to 28x28 fed into a hidden layer of 128 neurons operating on rectified linear unit activation funtions.
* The second part involved examining different approaches taken for EDA and data preparation for images being fed into deep learning models using openCV

## Assignment 2
The second assignment was also multi-part.
* The first part involved using the MNIST hand-written digets dataset and creating a CNN model for classification.
* The second part involved examining the CIFAR10 dataset (10 classes of various objects), performing preprocessing and creating a CNN model for classification. It then involves a cross-comparison of the custom CNN with two pre-trained models: VGG16 and RESNet50.

## Assignment 3
The third assignment involved exploring various image processing applications past simple classification problems.
* Firstly, I wrote code to perform segmentation masking on the oxford pet dataset.
* Then I cloned the mask RCNN racoon dataset and wrote code to extract border boxes from the images, and examined how border-box annoations work for training/testing data in unsupervised learning.
* Finally, I leveraged the pretrained YOLOv5 model to perform object classification and border-box annotations on some images.

## Assignment 4
The fourth assignment involved digging in-depth for some popular image processing and transformation algorithms.
* Image keypoints were extracted and compared between Scale Invariant Feature Transform (SIFT), Features from Accelerated Segment Test (FAST), and Oriented FAST and Rotated BRIEF (Orb)
* Point-matching output analysis is then written and exemplified on an image to show how rotation and object location/classification can be leveraged in image processing.
* Then, using existing facial-recognition labelled XML files, faceCascade allows border boxes to be drawn around real-time photos that can be taken via Google Colabs notebooks.
* Then parameter detection is used to classify and count objects - counting the number of apples in an image of an apple tree.

## Assignment 5
This assignment examines processing series of images in video data. It performs pose estimation from the video, and real-time video pose estimation, object classification, and object bounding boxes when enabled from Google Colabs.

## Assignment 6
This assignment explores generative models. It first exemplifies simple Generative Adversarial Networks (GANs) for digit generation from the MNIST dataset. It then builds a deep GANs model for building pokemon sprite images.
