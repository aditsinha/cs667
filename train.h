
#pragma once

#include <eigen3/Eigen/Dense>
#include "party.h"
#include "privacy.h"
#include <vector>

Eigen::VectorXd gradient_train_simulation(std::vector<Party*> p, Configuration* c);
Eigen::VectorXd train_single(Party* p, Configuration* c);
double evaluate_accuracy(Party* p, Eigen::VectorXd params);

double getLearningRate(Configuration* config, int iteration);
