
#include "party.h"

Configuration::Configuration(std::ifstream config_file)  {
  assert(config_file.is_open());

  // First read the type of training we are going to do
  config_file >> mode;
  // Next read the size of the dataset (number of parties, number of
  // entries per party, number of feature dimensions)
  config_file >> n >> m >> d;
  // For each feature dimension, read the normalization factor

  normalization = Eigen::VectorXd(d);

  for (int i = 0; i < d; i++) {
    std::string str;
    std::getline(config_file, str, ',');
    normalization(i) = std::stof(str);
  }
}

Party::Party(Configuration* config, std::ifstream data_file)
  : features(config->n, config->m)
{
  assert(data_file.is_open());

  n = config->n;
  m = config->m;
  d = config->d;

  features = Eigen::MatrixXd(m, d);
  labels = Eigen::VectorXd(m);
  
  // assume that data is in CSV format, where the first column is the
  // label
  
  for (int j = 0; j < m; j++) {
    std::string str;
    // read the label
    std::getline(data_file, str, ',');
    labels(j) = std::stoi(str);

    // now read the feature vector
    for (int k = 0; k < d; k++) {
      std::getline(data_file, str, ',');
      features(j,k) = std::stof(str) * config->normalization(k);
    }

    // Check that the label and features are correct
    assert(labels(j) == 0 || labels(j) == 1);
    assert(getFeatureVec(j).norm() <= 1);
  }
}

Eigen::VectorXd Party::getFeatureVec(int i) {
  return features.block(i,0,1,d);
}

double ComputeLogisticFn(Eigen::VectorXd params, Eigen::VectorXd features) {
  return 1 / (1 + exp(-1 * params.dot(features)));
}

Eigen::VectorXd Party::ComputeGradient(Eigen::VectorXd params) {
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(d);
  for (int i = 0; i < m; i++) {
    double error = ComputeLogisticFn(params, getFeatureVec(i)) - labels(i);
    grad += error * getFeatureVec(i);
  }

  return grad;
}
