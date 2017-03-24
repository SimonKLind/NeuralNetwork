# NeuralNetwork
A fully functioning neural network for image classification with CIFAR-10

This is my take on a Neural Network, refactored and rewritten from scratch.

Changes from First Attempt:
  - ReLU is not applied right after Fully Connected Layer, allowing BatchNorm to take its intended place
  - BatchNorm gradient is fixed
  - BatchNorm now has its learnable weight y and bias b, as intended
  - SoftMax gradient is fixed
  - Additional classes Matrix and Vector have been written to simplify a lot of the "heavy lifting"
  - Addition Net class has been made to make layers a bit more manageable
  - Weight initialization has been fixed
  
At its current state it gets a bit more than 40% on validation set with two layers, BatchNorm, ReLU, and SoftMax.

It's still not perfect though, according to the folks behind cs231n (The Stanford course I'm following to learn this stuff) you should be able to get over 50% with this setup.

TODO:
 - Load data into single 50000*3072 matrix instead of 5 matrices.
 - Normalize data (Subtract mean across each channel)
 - Redesign train function to make random mini-batches instead of using same batches over and over...
 - Add more layers
    - Dropout
    - Convolution (As soon as i find time to learn about it)
    - Pooling (As soon as i find time to learn about it)
