
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cmath>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "party.h"
#include "train.h"

int main(int argc, char** argv) {
  // the first argument is the configuration file
  // the second argument is the data file

  assert(argc == 3 || argc == 4);

  std::ifstream config_file(argv[1]);
  std::ifstream data_file(argv[2]);

  Configuration config(config_file);
  std::vector<Party*> parties;

  for (int i = 0; i < config.n; i++) {
    parties.push_back(new Party(&config, data_file, true));
  }

  std::ifstream validation_file;
  if (argc == 4) {
    validation_file = std::ifstream(argv[3]);
  } else {
    validation_file.swap(data_file);
  }

  Party validation(&config, validation_file, false);

  Eigen::VectorXd trained;
  trained = gradient_train_simulation(parties, &config);
  
  // std::cout << "Training Accuracy " << parties[0]->Accuracy(trained) << std::endl;
  std::cout << "Validation Accuracy " << validation.Accuracy(trained) << std::endl;
}
