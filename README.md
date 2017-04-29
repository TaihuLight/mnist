# mnist

Introduction
------------
A C++ implementation of a feed-forward neural network for training on the MNIST
database of handwritten digits. Intended as a private exercise to consolidating
some of the knowledge I gained from Andrew Ng's Machine Learning offering on
Coursera. Currently still a work in progress.

Instructions
------------
Run make all on the command line to build the project. Afterwards, an
executable `main` will have been created in `build/`. To run, first download
the MNIST database from
```
http://yann.lecun.com/exdb/mnist/
```
and extract the data files into the same directory. Next, run `main` with the
path to the MNIST data files as a command line argument. Currently, the
program will output its accuracy on the test set in a percentage for a fixed
choice of network parameters (e.g., the numbers of hidden layers and -nodes,
the learning rate, ...). In the future, these will be made configurable from
the command line.

TODO
----
Some of the changes still needed to be realized are as follows.
- Make the learning rate configurable from the command line.
- Add Doxygen documentation
- Switch Make for CMake
- Use Google Test for unit tests.
