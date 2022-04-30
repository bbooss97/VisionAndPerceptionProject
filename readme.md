The scope of this project is to realize a neural network able to gather from an image of a chess board his configuration.
The dataset that i want to use is https://www.kaggle.com/datasets/koryakinp/chess-positions
that contains a list of images of chess positions with a FEN description of them as label.
I would like to realize such a neural network that is able to get the position of all the pieces starting from an image modified with a random homography transformation with parameters bounded:
I want to evaluate the performance of the network on different bounds and see how the network behaves.
The network should be a classifier and i would like to use some pretrained networks in order to get better accuracy for example the ResNet to get the features and other layers to accomplish the task of classification.I may try different pretrained networks and different types of layers connected to the pretrained network or try to train a network from scratch.
In the end i would like to test the network on a real application like for example lichess and see if the network behaves correctely.
Andrea Morelli 1845525
