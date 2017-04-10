
#ifndef PARTY_H__
#define PARTY_H__

#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cmath>

#include <eigen3/Eigen/Dense>

class Configuration {
public:
  explicit Configuration(std::ifstream& config_file);
  int n, m, d;
  double clipping;
  Eigen::MatrixXd normalization;
  std::string mode;
  int batch_size;
  int fractional_bits;
  double epsilon;
  double delta;
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
  
  /* int n, m, d; */
  /* bool use_mini_batch; */
  /* int batch_size; */

  /* double should_clip_gradient; */
  /* double gradient_clip; */
};

#endif
