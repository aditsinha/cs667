
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
  Configuration(std::ifstream config_file);
  int n, m, d;
  Eigen::MatrixXd normalization;
  std::string mode;
};

class Party {
public:
  Party(Configuration* config, std::ifstream data_file);
  ~Party();
  float* ComputeGradient(float* params);

  Eigen::MatrixXd features;
  Eigen::MatrixXd labels;

  Eigen::VectorXd ComputeGradient(Eigen::VectorXd params);
  Eigen::VectorXd getFeatureVec(int i);
  
  int n, m, d;
};
