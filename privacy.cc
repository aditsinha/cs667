
#include "privacy.h"
#include "wrapper.h"

#include <random>
#include <cassert>
#include <cmath>
#include <iostream>

extern std::default_random_engine generator;

double sampleLaplace(double scale) {
  // choose magnitude from exponential distribution and direction uniformly at random
  std::exponential_distribution<double> exp_dist(1/scale);
  double magnitude = exp_dist(generator);
  int direction = (generator() % 2) ? 1 : -1;

  return magnitude * direction;
}

Eigen::VectorXd generateRandomDirection(int d) {
  std::normal_distribution<double> normal(0.0, 1.0);

  Eigen::VectorXd dir(d);
  for (int i = 0; i < d; i++) {
    dir(i) = normal(generator);
  }

  dir.normalize();
  return dir;
}

Eigen::VectorXd PrivacyParams::generateLaplaceNoise(double l1_sens, int d) {
  assert(epsilon > 0);
  
  Eigen::VectorXd noise_vec(d);

  for (int i = 0;i < d; i++) {
    noise_vec(i) = sampleLaplace(l1_sens / epsilon);
  }

  return noise_vec;
}

double PrivacyParams::getMomentsAccountStandardDev(double gradient_clip, int batch_size, int database_size, int num_epochs) {
  double l2_sens = 2 * gradient_clip / batch_size;
  double sample_prop = (double)batch_size / database_size;
  int num_steps = num_epochs * database_size / batch_size;

  double c2 = 1.485;
  return c2 * sample_prop * sqrt(num_steps * log(1 / delta) / epsilon) * l2_sens;
}


Eigen::VectorXd PrivacyParams::generateLogisticRegressionNoise(double gradient_clip, int batch_size, int database_size, int num_epochs, int d) {
  double sd = getMomentsAccountStandardDev(gradient_clip, batch_size, database_size, num_epochs);

  Eigen::VectorXd noise_vec(d);
  std::normal_distribution<double> normal(0.0, sd);
  for (int i = 0; i < d; i++) {
    noise_vec(i) = normal(generator);
  }

  return noise_vec;
}


PrivacyParams PrivacyParams::GetStepPrivacyParams(int repetitions, double step_delta) {
  double remain_delta = delta - repetitions*step_delta;

  assert(remain_delta >= 0);

  double step_epsilon = 0;
  if (epsilon > 0 && remain_delta > 0) {
    step_epsilon = epsilon / (2 * sqrt(2 * repetitions * log(1 / remain_delta)));
  } else if (epsilon > 0 && remain_delta == 0) {
    step_epsilon = epsilon / repetitions;
  }

  return PrivacyParams(step_epsilon, step_delta);
}

PrivacyParams PrivacyParams::GetSamplingPrivacyParams(int population_size, int sample_size) {
  double sample_prop = (double)sample_size / population_size;
  return PrivacyParams(sample_prop*epsilon, sample_prop*delta);
}
