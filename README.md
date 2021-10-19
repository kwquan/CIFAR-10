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

# Model Performance
Model is able to achieve an accuracy of 0.86, with crossentropy < 0.7 on the test set \
![alt text](https://github.com/kwquan/CIFAR-10/blob/main/model_accuracy.png)
![alt text](https://github.com/kwquan/CIFAR-10/blob/main/model_loss.png)

 # Common Questions
 1) Why is feature scaling required? \
 ![alt text](https://github.com/kwquan/CIFAR-10/blob/main/activations.png)
 
  Recall that we need to feed the inputs to activation functions, which essentially act as a map between the input data to outputs. Depending on which function you use, if input 
  data is too big, the derivatives of the functions w.r.t the inputs will be close to zero[see sigmoid,tanh]. \
  Referring to how weights are updated in my neural network from scratch example: \
  dSSR/dw1[Using chain rule] = dSSR/dPredicted * dPredicted/dy1 * dy1/dx1 * dx1/dw1, dy1/dx1 will be low which will then cause the whole product to be lowered. \
  As you can imagine, this will cause slower weight updates hence leading to slower convergence. \
  This is also known as the vanishing gradient problem. 
 
 2) Why do we need activation functions? \
This is to introduce non-linearity into the model instead of relying on layer outputs which are essentially linear transformations of the input data. This will better enable    the model to learn more complex patterns.
 
 3) Why Relu?
Relu is chosen to mitigate the vanishing gradient problem. Comparing the derivative equations for sigmoid & Relu, we will find that Relu will give much higher derivativesfor  values > 0. Again, using the weights update example above, multiplying a larger dy1/dx1 will mean that the entire product will decrease slower & thus mitigating the vanishing  gradient effect. Of course, we can use any other activation functions that has the same effect. 
 
 4) Why Batch Normalization?
Recall that everytime we feed a batch of data into training, we are essentially trying to map the inputs to the labels. This also means that our target predictions will keep shifting depending on the input batch, an effect also known as covariate shift. By normalizing the inputs, we help to mitigate this effect since all inputs will have the same distribution & hence have much closer target outputs for us to predict.  

