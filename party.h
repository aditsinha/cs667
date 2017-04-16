
#ifndef PARTY_H__
#define PARTY_H__

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include "privacy.h"

class Configuration {
public:
  explicit Configuration(std::ifstream& config_file);
  int n, m, d;
  double clipping;
  Eigen::MatrixXd normalization;
  std::string mode;
  int batch_size;
  int epochs;
  int fractional_bits;
  PrivacyParams privacy;
  float initial_learning_rate;
  float learning_rate_decay;
};

class Party {
public:
  Party(Configuration* config, std::ifstream& data_file);
  ~Party();
  double RMSE(Eigen::VectorXd params);
  double Accuracy(Eigen::VectorXd params);

  Eigen::MatrixXd features;
  Eigen::MatrixXd labels;

  Eigen::VectorXd MakePredictions(Eigen::VectorXd params);
  Eigen::VectorXd ComputeGradient(Configuration* config, Eigen::VectorXd params);
};

#endif
