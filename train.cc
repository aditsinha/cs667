
#include <eigen3/Eigen/Dense>

#include "party.h"
#include "train.h"

double getLearningRate(double initial, double decay, int epoch) {
  return initial / (1 + decay * epoch);
}

Eigen::VectorXd train_single(Party* p, Configuration* c) {
  Eigen::VectorXd params = Eigen::VectorXd::Zero(p->features.cols());

  float learning_rate = c->initial_learning_rate;

  int num_batches_per_epoch = c->m / c->batch_size;

  for (int i = 0; i < c->epochs; i++) {
    for (int j = 0; j < num_batches_per_epoch; j++) {
      auto gradient = p->ComputeGradient(c, params);

      gradient += c->privacy.generateLogisticRegressionNoise(c->clipping, c->batch_size, c->m, c->epochs, c->d);
      
      params = params - gradient * learning_rate;
    }

    std::cout << p->RMSE(params) << std::endl;
    learning_rate = getLearningRate(c->initial_learning_rate, c->learning_rate_decay, i);
  }

  return params;
}
