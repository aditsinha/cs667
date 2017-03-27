
#include <eigen3/Eigen/Dense>

#include "party.h"
#include "train.h"

double getLearningRate(double initial, double decay, int epoch) {
  return initial / (1 + decay * epoch);
}

Eigen::VectorXd train_single(Party* p, PrivacyParams pp, int quantize_bits) {
  Eigen::VectorXd params = Eigen::VectorXd::Zero(p->d);

  float initial_learning_rate = .25;
  float learning_rate_decay = 0.0001;

  int max_steps = 10000;
  int batch_size = 10;

  PrivacyParams step_privacy = pp.GetStepPrivacyParams(max_steps, 0);
  step_privacy = step_privacy.GetSamplingPrivacyParams(batch_size, p->m);

  for (int i = 0; i < max_steps; i++) {
    auto gradient = p->ComputeMiniBatchGradient(params, batch_size);
    gradient += generateLaplaceNoise(2.0 / p->m, p->d, step_privacy);
    // gradient = quantizeVector(gradient, quantize_bits);
    
    params = params - gradient * getLearningRate(initial_learning_rate, learning_rate_decay, i);

    if (i % 1000 == 0) {
      std::cout << p->RMSE(params) << std::endl;
    }
  }

  return params;
}
