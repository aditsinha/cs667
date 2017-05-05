
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
  int val_m;
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
  double TrainingAccuracy(Eigen::VectorXd params);
  double ValidationAccuracy(Eigen::VectorXd params);

  Eigen::MatrixXd features, val_features;
  Eigen::MatrixXd labels, val_labels;

  Eigen::VectorXd MakePredictions(Eigen::VectorXd params,
				  Eigen::MatrixXd target_features,
				  Eigen::VectorXd target_labels);
  Eigen::VectorXd ComputeGradient(Configuration* config, Eigen::VectorXd params);

private:
  double read_label(Configuration* config, std::ifstream& data_file);
  Eigen::VectorXd read_feature_row(Configuration* config, std::ifstream& data_file);

  double accuracy(Eigen::VectorXd params, Eigen::MatrixXd features, Eigen::VectorXd labels);
};

#endif
