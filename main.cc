
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

  assert(argc == 3);

  // For configuration file, first line contains n and m
  std::string config_filename = argv[1];


}
