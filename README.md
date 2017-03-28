# NeuralNetwork
A fully functioning neural network for image classification with CIFAR-10

This is my take on a Neural Network, refactored and rewritten from scratch.

# Update 1
With this update convergence is much faster and classification improved. The addition of randomizing mini-batches and normalizing the data allowed for higher classification accuracy after less than half of the training it did before.
I've only run this setup through 1000 batches once, where it reached a final classification accuracy of 50.11% over all 10000 validation images. In the same run the highest validation accuracy throughout training was 54.3%, during training it achieved a highest train accuracy of ~65%. This is with 2 fully connected layers with 200 neurons in the hidden layer, each layer followed by batch normalization and a ReLU, and finally SoftMax loss.

Changes:
 - Training data is now loaded into a single 50000x3072 matrix instead of 5 separate matrices
 - Data is now normalized
 - Training now trains on batches of randomly selected images, instead of cycling through same images over and over
 - After every batch the net checks validation accuracy with 1000 random images out of the 10000 validation images
 
# Update 0 
Changes from First Attempt:
  - ReLU is not applied right after Fully Connected Layer, allowing BatchNorm to take its intended place
  - BatchNorm gradient is fixed
  - BatchNorm now has its learnable weight y and bias b, as intended
  - SoftMax gradient is fixed
  - Additional classes Matrix and Vector have been written to simplify a lot of the "heavy lifting"
  - Addition Net class has been made to make layers a bit more manageable
  - Weight initialization has been fixed
  
At its current state it gets a bit more than 40% on validation set with two layers, BatchNorm, ReLU, and SoftMax.

It's still not perfect though, according to the folks behind cs231n (The Stanford course I'm following to learn this stuff) you should be able to get over 50% with this setup, with some hyperparameter tweaking.

TODO:
 - Add functionality to save Net params
 - Implement better parameter update function (e.g. adagrad or some other better method)
 - Add more layers
    - Dropout
    - Convolution (As soon as i find time to learn about it)
    - Pooling (As soon as i find time to learn about it)
