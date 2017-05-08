
#include "party.h"

#include <cassert>

#include <random>
#include <chrono>

std::default_random_engine generator;

Configuration::Configuration(std::ifstream& config_file) :
  privacy(0,0)
{
  assert(config_file.is_open());

  n = m = d = 0;
  clipping = 0;
  batch_size = 0;
  feature_scale = Eigen::VectorXd::Zero(d);

  auto seed_begin = std::chrono::high_resolution_clock::now();

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
    } else if (key == "num_validation_rows") {
      val_m = std::stoi(value);
    } else if (key == "num_dimensions") {
      d = std::stoi(value);
      feature_scale = Eigen::VectorXd::Ones(d);
    } else if (key == "feature_scale") {
      if (value.find(",") == std::string::npos) {
	// feature_scale is a single value
	feature_scale *= std::stof(value);
      } else {
	// feature_scale is a vector
	std::istringstream is_feature_scale(value);
	for (int i = 0; i < d; i++) {
	  std::string v;
	  std::getline(is_feature_scale, v, ',');
	  feature_scale(i) = std::stof(v);
	}
      }
    } else if (key == "gradient_clip") {
      clipping = std::stof(value);
    } else if (key == "batch_size") {
      batch_size = std::stoi(value);
    } else if (key == "fractional_bits") {
      fractional_bits = std::stoi(value);
    } else if (key == "privacy") {
      std::string v;
      std::istringstream s_privacy(value);
      std::getline(s_privacy, v, ',');
      privacy.epsilon = std::stof(v);
      std::getline(s_privacy, v, ',');
      privacy.delta = std::stof(v);

      std::cout << "Privacy: " << privacy.epsilon << ", " << privacy.delta << "\n";
      
    } else if (key == "epochs") {
      epochs = std::stoi(value);
    } else if (key == "initial_learning_rate") {
      initial_learning_rate = std::stof(value);
    } else if (key == "learning_rate_decay") {
      learning_rate_decay = std::stof(value);
    } else if (key == "regularization") {
      regularization = std::stof(value);
    }
  }

  auto seed_end = std::chrono::high_resolution_clock::now();
  generator.seed((seed_end - seed_begin).count());
}

Party::Party(Configuration* config, std::ifstream& data_file, bool is_training)
{
  assert(data_file.is_open());

  int num_rows = (is_training) ? config->m : config->val_m;

  features = Eigen::MatrixXd(num_rows, config->d);
  labels = Eigen::VectorXd(num_rows);

  // assume that data is in CSV format, where the first column is the
  // label
  for (int i = 0; i < num_rows; i++) {
    labels(i) = read_label(config, data_file);
    features.row(i) = read_feature_row(config, data_file);
    // Check the label
    assert(labels(i) == 0 || labels(i) == 1);
  }
}

double Party::read_label(Configuration* config, std::ifstream& data_file) {
  std::string str;
    // read the label
    std::getline(data_file, str, ',');
    return std::stoi(str);
}
  
Eigen::VectorXd Party::read_feature_row(Configuration* config, std::ifstream& data_file) {
  std::string str;
  // now read the feature vector
  Eigen::VectorXd row(config->d);
  for (int k = 0; k < config->d; k++) {
    std::getline(data_file, str, ',');
    row(k) = std::stof(str) * config->feature_scale(k);
  }

  return row;
}

double ComputeLogisticFn(Eigen::VectorXd params, Eigen::VectorXd features) {
  return 1 / (1 + exp(-1 * (params[0] + params.tail(features.size()).dot(features))));
}

Eigen::VectorXd Party::ComputeGradient(Configuration* config, Eigen::VectorXd params) {
  double p_include = (double)config->batch_size / config->m;

  std::uniform_real_distribution<double> urd;
  
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(config->d + 1);
  bool use_mini_batch = (config->batch_size > 0);
  bool should_clip_gradient = (config->clipping > 0);

#pragma omp parallel
  {
    Eigen::VectorXd lgrad = Eigen::VectorXd::Zero(config->d + 1);
    int num_included = 0;
    
#pragma omp for
    for (int i = 0; i < config->m; i++) {
      if (use_mini_batch && urd(generator) > p_include)  {
	continue;
      }
    
      double error = ComputeLogisticFn(params, features.row(i)) - labels(i);
      Eigen::VectorXd grad_i(config->d + 1);
      grad_i.tail(config->d) = error * features.row(i);
      grad_i(0) = error;
      double norm = grad_i.lpNorm<2>();
      if (should_clip_gradient && norm > config->clipping) {
	grad_i = grad_i * (config->clipping/norm);
      }

      lgrad += grad_i;
      num_included++;
    }

#pragma omp critical
    {
      grad += lgrad;
    }
  }

  grad /= config->m;

  return grad;
}

Eigen::VectorXd Party::MakePredictions(Eigen::VectorXd params, Eigen::MatrixXd target_features, Eigen::VectorXd target_labels) {
  int m = target_labels.rows();
  Eigen::VectorXd pred(m);
  for (int i = 0; i < m; i++) {
    pred(i) = ComputeLogisticFn(params, target_features.row(i));
  }

  return pred;
}

double Party::Accuracy(Eigen::VectorXd params) {
  int m = labels.rows(), errors = 0;
  Eigen::VectorXd preds = MakePredictions(params, features, labels);

  for (int i = 0; i < m; i++) {
    int int_pred = (preds(i) > .5) ? 1 : 0;
    errors += abs(int_pred - labels(i));
  }



  return 1 - (double)errors / m;
}

Party::~Party() {
  
}

