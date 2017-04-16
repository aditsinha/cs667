
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


int main(int argc, char* argv[]) {
  assert(argc == 4);
  std::ifstream config_file(argv[1]);
  std::ifstream data_file1(argv[2]);
  std::ifstream data_file2(argv[3]);

  Configuration config(config_file);
  Party p1(&config, data_file1), p2(&config, data_file2);

  load_sigmoid_taylor_coefficients();

  Eigen::MatrixXi p1_data = (p1.features * (1 << PRECISION)).cast<int>();
  Eigen::VectorXi p1_labels = (p1.labels).cast<int>();

  Eigen::MatrixXi p2_data = (p2.features * (1 << PRECISION)).cast<int>();
  Eigen::VectorXi p2_labels = (p2.labels).cast<int>();

  int iterations = config.epochs * config.m / config.batch_size;

  Eigen::VectorXi model = Eigen::VectorXi::Zero(config.d);

  int data_i = 0;
  for (int i = 0; i < iterations; i++) {
    Eigen::VectorXi gradient = Eigen::VectorXi::Zero(config.d);
    
    for (int j = 0; j < config.batch_size; j++) {
      auto p1_row = Eigen::VectorXi(p1_data.row(data_i));
      auto p2_row = Eigen::VectorXi(p2_data.row(data_i));
      
      add_to_gradient(gradient.data(), model.data(), p1_row.data(), p1_labels(data_i), config.d);
      add_to_gradient(gradient.data(), model.data(), p2_row.data(), p2_labels(data_i), config.d);

      data_i = (data_i + 1) % config.m;
    }

    int learning_rate = getLearningRate(&config, i) * (1 << PRECISION);
    mult_ovec_p(gradient.data(), -learning_rate / config.batch_size / 2, gradient.size());
    add_ovecs(model.data(), gradient.data(), model.size());
  }
  
  Eigen::VectorXd final_model = model.cast<double>() / (1 << PRECISION);

  std::cout << "Party 1 Accuracy: " << p1.Accuracy(final_model) << std::endl;
  std::cout << "Party 2 Accuracy: " << p2.Accuracy(final_model) << std::endl;
}
