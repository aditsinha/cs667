
#include "party.h"

#include <random>

std::default_random_engine generator;

Configuration::Configuration(std::ifstream& config_file)  {
  assert(config_file.is_open());

  n = m = d = 0;
  clipping = 0;
  batch_size = 0;

  std::string line;
  while (std::getline(config_file, line)) {
    std::istringstream is_line(line);
    std::string key, value;
    std::getline(is_line, key, '=');
    std::getline(is_line, value);

    if (key == "noise") {
      mode = value;
    } else if (key == "num_parties") {
      n = std::stoi(value);
    } else if (key == "num_data_rows") {
      m = std::stoi(value);
    } else if (key == "num_dimensions") {
      d = std::stoi(value);
    } else if (key == "normalization") {
      normalization = Eigen::VectorXd(d);
      std::istringstream is_normalization(value);
      for (int i = 0; i < d; i++) {
	std::string v;
	std::getline(is_normalization, v, ',');
	normalization(i) = std::stof(v);
      }
    } else if (key == "gradient_clip") {
      clipping = std::stof(value);
    } else if (key == "batch_size") {
      batch_size = std::stoi(value);
    } else if (key == "fractional_bits") {
      fractional_bits = std::stoi(value);
    } else if (key == "epsilon") {
      epsilon = std::stof(value);
    } else if (key == "delta") {
      delta = std::stof(value);
    }
  }
}

Party::Party(Configuration* config, std::ifstream& data_file)
  : features(config->n, config->m)
{
  assert(data_file.is_open());

  features = Eigen::MatrixXd(config->m, config->d);
  labels = Eigen::VectorXd(config->m);
  
  // assume that data is in CSV format, where the first column is the
  // label
  
  for (int j = 0; j < config->m; j++) {
    std::string str;
    // read the label
    std::getline(data_file, str, ',');
    labels(j) = std::stoi(str);

    // now read the feature vector
    for (int k = 0; k < config->d; k++) {
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

Eigen::VectorXd Party::ComputeGradient(Configuration* config, Eigen::VectorXd params) {
  double p_include = (double)config->batch_size / config->m;
  int num_included = 0;
  std::uniform_real_distribution<double> urd;
  
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(config->d);
  bool use_mini_batch = (config->batch_size > 0);
  bool should_clip_gradient = (config->clipping > 0);
  
  for (int i = 0; i < config->m; i++) {
    if (use_mini_batch && urd(generator) > p_include)  {
      continue;
    }
    
    double error = ComputeLogisticFn(params, features.row(i)) - labels(i);
    Eigen::VectorXd grad_i = error * features.row(i);
    double l1_norm = grad_i.lpNorm<2>();
    if (should_clip_gradient && l1_norm > config->clipping) {
      grad_i = grad_i * (config->clipping/l1_norm);
    }

    grad += grad_i;
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
