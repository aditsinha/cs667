
#include <iostream>
#include <fstream>
#include <cassert>

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
  assert(argc == 4);
  std::ifstream config_file(argv[1]);
  std::ifstream data_file1(argv[2]);
  std::ifstream data_file2(argv[3]);

  Configuration config(config_file);
  Party p1(&config, data_file1), p2(&config, data_file2);

  load_sigmoid_taylor_coefficients();

  MatrixXl p1_data = (p1.features * (1L << PRECISION)).cast<long>();
  VectorXl p1_labels = (p1.labels).cast<long>();

  MatrixXl p2_data = (p2.features * (1L << PRECISION)).cast<long>();
  VectorXl p2_labels = (p2.labels).cast<long>();

  int iterations = config.epochs * config.m / config.batch_size;

  VectorXl model = VectorXl::Zero(config.d);

  // std::cout << VectorXl(p1_data.row(0)) << std::endl;

  int data_i = 0;
  for (int i = 0; i < iterations; i++) {
    VectorXl gradient = VectorXl::Zero(config.d);

    Eigen::VectorXd dmodel = model.cast<double>() / (1UL << PRECISION);
    auto dgrad = p1.ComputeGradient(&config, dmodel) + p2.ComputeGradient(&config, dmodel);

    for (int j = 0; j < config.batch_size; j++) {
      auto p1_row = VectorXl(p1_data.row(data_i));
      auto p2_row = VectorXl(p2_data.row(data_i));
      
      add_to_gradient(gradient.data(), model.data(), p1_row.data(), p1_labels(data_i), config.d);
      add_to_gradient(gradient.data(), model.data(), p2_row.data(), p2_labels(data_i), config.d);

      data_i = (data_i + 1) % config.m;
    }

    auto idgrad = (gradient.cast<double>()) / (1L << PRECISION);
    std::cout << idgrad << std::endl << std::endl;

    Eigen::MatrixXd gm(config.d, 2);
    gm.col(0) = idgrad;
    gm.col(1) = dgrad;

    std::cout << "Gradients: " << std::endl;
    std::cout << gm << std::endl;

    std::cout << "Gradient Error: " << (idgrad - dgrad).norm() << std::endl << std::endl;

    long learning_rate = getLearningRate(&config, i) * (1UL << PRECISION);
    mult_ovec_p(gradient.data(), -learning_rate / config.batch_size / 2, gradient.size());
    add_ovecs(model.data(), gradient.data(), model.size());
  }
  
  Eigen::VectorXd final_model = model.cast<double>() / (1L << PRECISION);

  std::cout << "Party 1 Accuracy: " << p1.Accuracy(final_model) << std::endl;
  std::cout << "Party 2 Accuracy: " << p2.Accuracy(final_model) << std::endl;
}
