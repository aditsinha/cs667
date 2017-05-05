
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

  std::ifstream config_file(argv[1]);
  std::ifstream data_file(argv[2]);

  Configuration config(config_file);
  std::vector<Party*> parties;

  for (int i = 0; i < config.n; i++) {
    parties.push_back(new Party(&config, data_file));
  }

  Eigen::VectorXd trained;
  if (config.n == 1) {
    trained = train_single(parties[0], &config);
  } else {
    trained = gradient_train_simulation(parties, &config);
  }
  
  std::cout << "Training Accuracy " << parties[0]->TrainingAccuracy(trained) << std::endl;
  std::cout << "Validation Accuracy " << parties[0]->ValidationAccuracy(trained) << std::endl;
}
