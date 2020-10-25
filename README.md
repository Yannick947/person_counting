# Person Counting

This repository aims to train a convolutional-recurrent neural network to count persons in videos. The algorithm is trained by using the [pcds dataset](https://github.com/shijieS/people-counting-dataset) to count persons passing a bus entrance.

The architecture of the best algorithm trained so far can be seen in the following figure: 
![alt-text-1](/images/best_convoultion_recurrent_architecture.png "Best architecture for the convolution recurrent neural network")


## Train and evaluate in Google Colab 

The training logging and further usage with the person counting algorithm was optimized to train in the Google Colaboratory environment. Mounting this folder in The notebooks provided can be directly used to train the algorithms and the logging and saving of the weights will be done in the related Google Drive folder. 

A Google Colab notebook is provided which can be found in the colab_notebooks folder with the name trainer_counting to train the algorithm. In the /bin/cnn_regression.py file you can find the hyperparameter space to tune the architecture or learning parameters. The results will be automatically logged if you mount a Google Drive before starting the training session. 
Tensorboard logging is activated so you can see live performances of your models on your local machine. To get the results on your local machine you have to activate the automatic synchronization with your local machine from the Google Drive.

## Input Data

Detection Frames are the inputs for the person counting algorithm. You can create those .npy files for every video in a specified folder with the [person detection](https://github.com/Yannick947/person_detection) repository. All x- and y-coordinates of person centers which were detected by a specified RetinaNet model will be saved in a .npy file. You can load this .npy file into a numpy array. An example detection frame for x- y- and time-coordinate can be seen here: 

<p float="left">
  <img src="images/entering_persons_x_t_coordinate.png" width="200" />
  <img src="images/entering_persons_y_t_coordinate.png" width="200" /> 
</p>


## Further ideas: 

Inspired by the 3D input data (x-, y- and time coordinates from the video) in the image below, a 3D convolution Kernel would be an interesting approach.  

![alt-text-1](/images/8persons_3d_plot.png "8 Persons entering the bus for x- y- and t-coordinate")

