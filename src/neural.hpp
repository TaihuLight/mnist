#ifndef NEURAL_HPP_
#define NEURAL_HPP_

#include <cmath>
#include <stdexcept>
#include <cstdint>

#include <armadillo>

namespace mnist {

class NeuralNet {
public:
  explicit          NeuralNet();
                    NeuralNet(const NeuralNet &) = delete;
                    NeuralNet(NeuralNet &&) = delete;
  NeuralNet&        operator=(const NeuralNet &) = delete;
  NeuralNet&        operator=(NeuralNet&&) = delete;
  void              LearnWeights(const arma::Mat<uint8_t>&,
                        const double = 0.015, const double = 0.095, int = 20);
  double            Evaluate(const arma::Mat<uint8_t>&) const;
private:
  enum {
    kBatchSz = 50,
    kInputLayerSz = 784,
    kHiddenLayerSz = 30,
    kOutputLayerSz = 10,
    kWeightsHeadSz = kHiddenLayerSz * (kInputLayerSz + 1),  // rows * cols
    kWeightsTailSz = kOutputLayerSz * (kHiddenLayerSz + 1), // rows * cols
    kWeightsSz = kWeightsHeadSz + kWeightsTailSz
  };
  arma::vec         weights_;
  mutable arma::mat activ_l1_;
  mutable arma::mat activ_l2_;
  mutable arma::mat activ_l3_;
  static void       ValidateSize(const arma::Mat<uint8_t>&);
  void              InitWeights();          // randomly initializes weights
  void              ForwardProp(const arma::Mat<uint8_t>&) const;
  arma::vec         BackProp(const arma::Col<uint8_t>&, const double) const;
};

inline NeuralNet::NeuralNet()
  : weights_(kWeightsSz)
  , activ_l1_(kBatchSz, kInputLayerSz + 1)
  , activ_l2_(kBatchSz, kHiddenLayerSz + 1)
  , activ_l3_(kBatchSz, kOutputLayerSz)
{
  // set bias activations
  activ_l1_.col(0).ones();
  activ_l2_.col(0).ones();
}

inline void NeuralNet::ValidateSize(const arma::Mat<uint8_t>& data)
{
  // Training- and test set sizes known from the MNIST database
  if (data.n_rows % kBatchSz != 0) {
    throw std::runtime_error{"Unexpected dimensions of input data"};
  }
}

} // namespace mnist

#endif // NEURAL_HPP_
