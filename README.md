# NeuralNetwork
A fully functioning neural network for image classification with CIFAR-10

This is my take on a Neural Network, refactored and rewritten from scratch.

# Update 2
The parameter updates have now been changed to use better method, as of right now I can't decide wheter to use RMSprop or Adam Update. As a result both methods are now supported and which one is used depends on preprocessor definitions. Adam Update seems to be a bit more stable in its descent, but RMSprop seems to result in a ~0.5% increase in performance on the final validation. A dropout layer has now been added as well, though I had to experiment a bit in order for it to actually improve the network. It seems that either my dropout implementation is a bit off, or it takes more training to get good results. Training a dropout net from scratch over the same amount of mini-batches as a non-dropout net resulted in a ~4% decrease in performance relative to the non-dropout net (though this was with only 1000 batches). However after training the net without dropout for the first half of batches and then fine-tuning it with dropout over the last half i got a ~3% increase, so the same two-layered net with 200 hidden neurons from the previous update now got 52.9% over all 10000 test images.

Changes:
 - Updates now use either RMSprop or Adam Update, can be switched with preprocessor definitions
 - Dropout layer added

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

# TODO:
 - COMMENT
 - Add functionality to save Net params
 - Add more layers
    - Convolution (As soon as i find time to learn about it)
    - Pooling (As soon as i find time to learn about it)
