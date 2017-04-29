#include <cassert>
#include <iostream>
#include <string>

#include "img_parser.hpp"
#include "lab_parser.hpp"

#include "neural.hpp" // TODO

using std::cout;
using std::string;

using arma::Mat;

using mnist::ImageParser;
using mnist::LabelParser;
using mnist::NeuralNet;

enum {
  kTrainingSetSz = 60000,
  kTestSetSz = 10000,
  kImageSz = 784
};

// MNIST image- and label files
static const string kTrainingSetImageFile = "train-images.idx3-ubyte";
static const string kTrainingSetLabelFile = "train-labels.idx1-ubyte";
static const string kTestSetImageFile     = "t10k-images.idx3-ubyte";
static const string kTestSetLabelFile     = "t10k-labels.idx1-ubyte";

int main(int argc, char *argv[])
{
  if (argc != 2) {
    cout << "Usage: main <directory>\n"; // TODO
    return 1;
  }

  // Training set
  Mat<uint8_t> training_set(kTrainingSetSz, kImageSz + 1);

  ImageParser img_train_parser{(argv[1] + kTrainingSetImageFile).c_str()};
  training_set.head_cols(kImageSz) = img_train_parser.Parse();

  LabelParser lab_train_parser{(argv[1] + kTrainingSetLabelFile).c_str()};
  training_set.tail_cols(1) = lab_train_parser.Parse();

  // Test set
  Mat<uint8_t> test_set(kTestSetSz, kImageSz + 1);

  ImageParser img_test_parser{(argv[1] + kTestSetImageFile).c_str()};
  test_set.head_cols(kImageSz) = img_test_parser.Parse();

  LabelParser lab_test_parser{(argv[1] + kTestSetLabelFile).c_str()};
  test_set.tail_cols(1) = lab_test_parser.Parse();

  // Create neural network
  NeuralNet nn{};

  // Learn weights from training set and output cost
  nn.LearnWeights(training_set);

  // Evaluate test set and output cost
  cout << nn.Evaluate(test_set) << '\n';

  return 0;
}
