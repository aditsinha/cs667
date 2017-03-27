
#include "party.h"

std::default_random_engine generator;

Configuration::Configuration(std::ifstream& config_file)  {
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

Party::Party(Configuration* config, std::ifstream& data_file)
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
    assert(features.row(j).norm() <= 1);
  }
}

double ComputeLogisticFn(Eigen::VectorXd params, Eigen::VectorXd features) {
  return 1 / (1 + exp(-1 * params.dot(features)));
}

Eigen::VectorXd Party::ComputeBatchGradient(Eigen::VectorXd params) {
  return ComputeMiniBatchGradient(params, m);
}

Eigen::VectorXd Party::ComputeMiniBatchGradient(Eigen::VectorXd params, int exp_batch_size) {
  double p_include = (double)exp_batch_size / m;
  int num_included = 0;
  std::uniform_real_distribution<double> urd;
  
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(d);
  for (int i = 0; i < m; i++) {
    if (urd(generator) > p_include)  {
      continue;
    }
    
    double error = ComputeLogisticFn(params, features.row(i)) - labels(i);
    grad += error * features.row(i);
    num_included++;
  }
  assert(num_included > 0);

  grad /= num_included;

  return grad;
}

Eigen::VectorXd Party::MakePredictions(Eigen::VectorXd params) {
  Eigen::VectorXd pred(m);
  for (int i = 0; i < m; i++) {
    pred(i) = ComputeLogisticFn(params, features.row(i));
  }

  return pred;
}

double Party::RMSE(Eigen::VectorXd params) {
  double error = 0;
  Eigen::VectorXd preds = MakePredictions(params);
  
  for (int i = 0; i < m; i++) {
    error += pow(labels(i) - preds(i), 2);
  }

  return sqrt(error / m);
}

double Party::Accuracy(Eigen::VectorXd params) {
  int errors = 0;
  Eigen::VectorXd preds = MakePredictions(params);

  for (int i = 0; i < m; i++) {
    int int_pred = (preds(i) > .5) ? 1 : 0;
    errors += abs(int_pred - labels(i));
  }

  return 1 - (double)errors / m;
}

Party::~Party() {
  
}
