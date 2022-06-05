# Convolutional-Networks

In exercise 2, we have implemented a convolutional neural network to perform image classification and explore
methods to improve the training performance and generalization of these networks.

We have used the CIFAR-10 dataset as a benchmark for our networks similar to the previous exercise. This dataset
consists of 50000 training images of 32x32 resolution with 10 object classes, namely airplanes, cars, birds, cats,
deer, dogs, frogs, horses, ships, and trucks. The task is to implement convolutional network to classify these
images using the PyTorch library. The four questions are

• Implement the ConvNet, train it and visualizing its weights(Question 1).

• Experiment with batch normalization and early stopping (Question 2).

• Data augmentation and dropout to improve generalization (Question 3).

• Implement transfer learning from a ImageNet pretrained model. (Question 4).

Questions 1-3 are based on the script ex3_convnet.py and question 4 is based on the script ex3_pretrained.py.
To download the CIFAR-10 dataset execute the script datasets/get datasets.sh or set the download flag to
True in the torchvision.datasets.CIFAR10 function call.
