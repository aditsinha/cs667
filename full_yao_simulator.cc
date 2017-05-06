
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>

#include "party.h"
#include "train.h"
#include "obliv_math_func.h"

extern "C"
{
#include "obliv_math_def.h"
}

typedef Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> MatrixXl;
typedef Eigen::Matrix<long, Eigen::Dynamic, 1> VectorXl;

int main(int argc, char* argv[]) {
  assert(argc >= 3);
  std::ifstream config_file(argv[1]);
  std::ifstream data_file1(argv[2]);

  Configuration config(config_file);
  Party p1(&config, data_file1, true);
  
  std::ifstream data_file2;
  if (argc >= 4) {
    data_file2 = std::ifstream(argv[3]);
  } else {
    data_file2.swap(data_file1);
  }

  Party p2(&config, data_file2, true);

  std::ifstream validation_file;
  if (argc == 5) {
    validation_file = std::ifstream(argv[4]);
  } else {
    validation_file.swap(data_file2);
  }

  Party val_party(&config, validation_file, false);

  load_sigmoid_taylor_coefficients();

  MatrixXl p1_data = (p1.features * (1L << PRECISION)).cast<long>();
  VectorXl p1_labels = (p1.labels).cast<long>();

  MatrixXl p2_data = (p2.features * (1L << PRECISION)).cast<long>();
  VectorXl p2_labels = (p2.labels).cast<long>();

  int iterations = config.epochs * config.m / config.batch_size;

  VectorXl model = VectorXl::Zero(config.d+1);

  long regularization = config.regularization * (1L << PRECISION);

  auto t1 = std::chrono::high_resolution_clock::now();

  int data_i = 0;
  for (int i = 0; i < iterations; i++) {
    std::cout << "Step: " << i << std::endl;
    VectorXl gradient = VectorXl::Zero(config.d+1);

    for (int j = 0; j < config.batch_size; j++) {
      auto p1_row = VectorXl(p1_data.row(data_i));
      auto p2_row = VectorXl(p2_data.row(data_i));
      
      add_to_gradient(gradient.data(), model.data(), p1_row.data(), p1_labels(data_i), config.d);
      add_to_gradient(gradient.data(), model.data(), p2_row.data(), p2_labels(data_i), config.d);

      data_i = (data_i + 1) % config.m;
    }

    add_regularization(gradient.data(), model.data(), regularization, config.d);

    long learning_rate = getLearningRate(&config, i) * (1L << PRECISION);
    mult_ovec_p(gradient.data(), -learning_rate / config.batch_size / 2, gradient.size());
    add_ovecs(model.data(), gradient.data(), model.size());
  }

  auto noise_vec =
    generate_noise_vec(config.d+1,
		       config.privacy.getRegularizedRegressionStandardDev(config.regularization, 2*config.m),
		       config.fractional_bits);

  add_ovecs(model.data(), noise_vec, model.size());
  
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "Elapsed Time: " <<
    std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms\n";
  
  Eigen::VectorXd final_model = model.cast<double>() / (1L << PRECISION);

  std::cout << "Party 1 Training Accuracy: " << p1.Accuracy(final_model) << std::endl;
  std::cout << "Party 2 Training Accuracy: " << p2.Accuracy(final_model) << std::endl;
  std::cout << "Validation Accuracy: " << val_party.Accuracy(final_model) << std::endl;
}
