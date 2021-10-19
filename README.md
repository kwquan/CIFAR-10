# CIFAR-10 Image Classification
In this notebook, we shall build a convolutional neural network using Keras in order to classify images from the CIFAR-10 dataset. \
This dataset consists of 10 classes with 6000 images each. \
Furthermore, it is split into training set[50000] & test set[10000], therefore having a total of 60000 images. 

# Dataset
Images are of the dimension[32 x 32] with 3 channels[RGB]: \
![alt text](https://github.com/kwquan/CIFAR-10/blob/main/sample.png)

# Network Architecture
![alt text](https://github.com/kwquan/CIFAR-10/blob/main/cifar_10_nn.drawio.png)

The whole model consists of 18 layers: 
1) Convolution, 32 filters with size (3x3) [Relu activation with Batch Normalization] 
2) Convolution, 32 filters with size (3x3) [Relu activation with Batch Normalization] 
3) Convolution, 32 filters with size (3x3) [Relu activation with Batch Normalization] 
4) Max Pool with size (2x2) 
5) Dropout (0.5) 
6) Convolution, 64 filters with size (3x3) [Relu activation with Batch Normalization] 
7) Convolution, 64 filters with size (3x3) [Relu activation with Batch Normalization] 
8) Convolution, 64 filters with size (3x3) [Relu activation with Batch Normalization] 
9) Max Pool with size (2x2) 
10) Dropout (0.5) 
11) Convolution, 128 filters with size (3x3) [Relu activation with Batch Normalization] 
12) Convolution, 128 filters with size (3x3) [Relu activation with Batch Normalization] 
13) Convolution, 128 filters with size (3x3) [Relu activation with Batch Normalization] 
14) Max Pool with size (2x2) 
15) Dropout (0.5) 
16) Dense (units = 32) [Relu activation] 
17) Dropout (0.5) 
18) Dense (units = 10) [Softmax activation]
 

