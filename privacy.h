
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

  Eigen::VectorXd generateLaplaceNoise(double l1_sens, int d);
  Eigen::VectorXd generateMomentsAccountNoise(double l2_sens, double sample_prop, int num_steps, int d);

  double epsilon, delta;
};

vector_t* quantizeVector(Eigen::VectorXd vec, int bits);
Eigen::VectorXd unquantizeVector(vector_t* vec, int bits);

Eigen::VectorXd generateLaplaceNoise(double l1_sens, int d, PrivacyParams pp);
