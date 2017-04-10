
#include <eigen3/Eigen/Dense>

#include "party.h"
#include "train.h"

double getLearningRate(double initial, double decay, int epoch) {
  return initial / (1 + decay * epoch);
}

Eigen::VectorXd train_single(Party* p, PrivacyParams pp, int quantize_bits) {
  Eigen::VectorXd params = Eigen::VectorXd::Random(p->d);

  float initial_learning_rate = .1;
  float learning_rate_decay = 0.1;

  int total_epochs = 100;
  int batch_size = 100;

  int num_batches_per_epoch = p->m / batch_size;
  double sample_proportion = (double)batch_size / p->m;

  double gradient_clip = .2;

  for (int i = 0; i < total_epochs; i++) {
    for (int j = 0; j < num_batches_per_epoch; j++) {
      auto gradient = p->ComputeMiniBatchGradient(params, batch_size, gradient_clip);
      gradient += pp.generateMomentsAccountNoise(2 * gradient_clip / batch_size, sample_proportion,
      						 total_epochs * num_batches_per_epoch, p->d);
      
      // gradient = quantizeVector(gradient, quantize_bits);
      params = params - gradient * getLearningRate(initial_learning_rate, learning_rate_decay, i);
    }

    std::cout << p->RMSE(params) << std::endl;
  }

  return params;
}
