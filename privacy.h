
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

  Eigen::VectorXd generateGaussianNoise(double sd, int dim);
  Eigen::VectorXd generateLaplaceNoise(double l1_sens, int d);
  Eigen::VectorXd generateLogisticRegressionNoise(double gradient_clip, int batch_size, int population_size, int num_epochs, int d);
  Eigen::VectorXd generateLogisticRegressionMPCNoise(double gradient_clip, int batch_size, int population_size, int num_epochs, int d, int n);
  double getMomentsAccountStandardDev(double gradient_clip, int batch_size, int database_size, int num_epochs);
  double getRegularizedRegressionStandardDev(double regularization, int population_size);
  double epsilon, delta;
};

Eigen::VectorXd generateLaplaceNoise(double l1_sens, int d, PrivacyParams pp);
Eigen::VectorXd reduceVectorPrecision(Eigen::VectorXd in, int bits);
