
#pragma once

#include <eigen3/Eigen/Dense>

struct PrivacyParams {
  PrivacyParams(double e, double d) {
    epsilon = e;
    delta = d;
  }

  // Returns the privacy parameters necessary per step to guarantee
  // this level of privacy when making the provided number fo steps
  PrivacyParams GetStepPrivacyParams(int repetitions, double step_delta);

  // Returns the privacy params that results from using a mechanism
  // with this privacy over a random sample
  PrivacyParams GetSamplingPrivacyParams(int population_size, int sample_size);

  double epsilon, delta;
};

Eigen::VectorXd quantizeVector(Eigen::VectorXd old, int bits);
Eigen::VectorXd generateLaplaceNoise(double l1_sens, int d, PrivacyParams pp);
