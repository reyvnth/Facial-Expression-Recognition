   #                                                        Real time Facial Expression Recognition

### INTRODUCTION

According to reports from various web sources, the most widely used mode of communication used by humans is Facial expression. Facial expression recognition is an evolving technology in the field of human-computer interaction. Facial expression recognition has  its  branches  spread  across  various  applications  such  as  virtual  reality,  webinar  technologies,  online  surveys  and  many  other fields.  Even though high advancements have been witnessed in this field, there are several diplomacies that exist. The traditional feature extraction methods have slower response and lack in performance. The traditional methods have high latency or delay in their response. Through these traditional methods, it is extremely difficult to extract the required features effectively and hence, is too hazardous to utilize for real time applications.  In order to provide a panacea to such issues, a facial recognition method is proposed using CNN. This model thus, can be used to solve the above stated problems or difficulties. The development of the facial expression recognition is done in Keras. A six layered CNN is developed to build and train the model. Each layer is defined with certain training techniques to enable faster and efficient feature extraction. The trained model is deployed to a web interface using Flask app. The developed model is applied for real time videos and images and its accuracy is analyzed. 

### OBJECTIVES OF THE PROJECT

The objectives of the project are as follows:
• To explore the dataset FER-2013.
• To generate training and validation batches. 
• To create a Convolutional Neural Network (CNN) model.
• To train and evaluate model 
• To represent the model as JSON string
• To create a Flask app to serve predictions
• To design an HTML template for the Flask app 
• Use the model to recognize facial expressions in real time and analyze its accuracy. 

### FER-2013 DATASET AND ITS FEATURES 
In this project, the dataset used to train the models is FER-2013. The FER-2013 dataset consists of 35887 images, of which 28709 labelled images belong to the training set and the remaining 7178 images belong to the test set. The images in FER-2013 dataset is labeled as  one  of the  seven  universal  emotions:  Happy,  Sad,  Angry,  Fear,  Surprise,  Disgust, and  Neutral.  Among these  emotion classifications, the most images belong to ‘happy’ emotions account to 7215 images of the 28709 training images. The number of images that belong to each emotion is given by returning the length of each directory using the OS module in Python. The images in FER-2013 dataset are in grayscale and is of dimensions 48x48 pixels. This dataset was created by gathering results from Google Image search of each emotions. The number of images of each emotion is given in table 1. The number of images of each emotion type is returned by the functions of ‘OS’ module in python. To explore the dataset further, and to understand what kind of images lie in the dataset, we plot few example images from the dataset using the ‘utils’ module in python. The resultant plot of example images obtained is given in figure 1. From various research papers, we have studied that the average attainable accuracy of a training model developed using FER-2013 dataset is 66.7% and our aim is also to design a CNN model with similar or a better accuracy.  

![download1](https://user-images.githubusercontent.com/64673748/96959154-fa726280-151c-11eb-819b-afb3b57c1fe0.png)

### TRAINING AND VALIDATION
During training, to minimize the losses of Neural Network, an algorithm called Mini-Batch Gradient Descent has been used. Mini-Batch Descent is a type of Gradient Descent algorithm used for finding the weights or co-efficient of artificial neural networks by splitting the training dataset into small batches. This algorithm provides more efficiency whilst training data.  

#### Generating Training and Validation Batches 
To build the training model, the image size and batch size are initialized. The image size is set as per the sizes of the images of the dataset. The batch size is set a value according to the memory requirements of the CPU or GPU hardware, so as to speed up the training process. The ImageDataGenerator() class of Keras library in python is used to accept values or images to train. The images in the dataset are flipped horizontally, as the real time images obtained through camera will be a mirrored image. These images are then used to generate a training set. The parameters such as target size, color mode, batch size, class mode and shuffle are given as per the requirements. Similarly, test set is generated with same parametric values as that of training set. 

#### CNN Model 
The CNN designed is based on sequential model and is designed to have six activation layers, of which 4 are convolutional layers and the remaining 2 are fully controlled layers.  The  4  convolutional  layers  consists  of  similar  training  techniques  such  as  Batch  Normalization,  ReLu  Activation  function, Maxpooling and Dropout. Batch Normalization technique is used as it trains networks faster, allows higher learning rates, simplifies deeper networks, and also makes weights easier to initialize. The ReLu activation function is used to increase non-linearity in the images and also makes evaluation quicker. Maxpooling is done to reduce the dimensionality of an image, i.e. reduce its height and width.  Dropout  function  is  added  to  the  layer  to  prevent  overfitting  of  the  training  data. The  fourth  convolutional  layer  has  an additional training technique called Flatten. The Flatten function converts the image into a 1-Dimensional array. This then, outputs the 1-Dimensional array as input for the Fully Controlled layers.  The two fully controlled layers are designed with training techniques like Dense, Batch Normalization, ReLu activation function and Dropout. The Dense function is used to connect each neuron of the previous layer to each neuron of the next layer. The remaining functions have similar applications. The output layer has two training techniques, Dense and softmax. The softmax function outputs a vector that returns the probability distributions of a list of potential outcomes.   The  volume  of  each  successive layer becomes half  of  its  previous layer  and  the  number  of channels  doubles  for each  successive layer. The below given figure 2 represents the structure of our CNN model. 

![download (1)1](https://user-images.githubusercontent.com/64673748/96960089-1971f400-151f-11eb-8794-0e19025b6acd.png)


[To Read the entire document click here....](https://www.researchgate.net/publication/342107269_Real-time_facial_expression_recognition_using_CNN)

DOWNLOAD all the files and save all files in a single folder.  

DOWNLOAD THE FER-2013 Dataset and save in the same folder.folder 

Similarly, change directory of the utils python to this .

run main.py by selecting the root file in the files section of python.
