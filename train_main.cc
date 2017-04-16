
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include "party.h"
#include "train.h"

int main(int argc, char** argv) {
  // the first argument is the configuration file
  // the second argument is the data file

  std::ifstream config_file(argv[1]);
  std::ifstream data_file(argv[2]);

  Configuration config(config_file);
  Party p(&config, data_file);

  auto trained = train_single(&p, &config);

  std::cout << "Accuracy " << p.Accuracy(trained) << std::endl;
}
