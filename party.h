
#pragma once

#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cmath>

#include <eigen3/Eigen/Dense>

class Configuration {
public:
  Configuration(std::ifstream& config_file);
  int n, m, d;
  Eigen::MatrixXd normalization;
  std::string mode;
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
  Eigen::VectorXd ComputeBatchGradient(Eigen::VectorXd params);
  Eigen::VectorXd ComputeMiniBatchGradient(Eigen::VectorXd params, int exp_batch_size);
  
  int n, m, d;
};
