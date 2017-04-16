
#pragma once

#include <eigen3/Eigen/Dense>
#include "party.h"
#include "privacy.h"

Eigen::VectorXd train_single(Party* p, Configuration* c);
double evaluate_accuracy(Party* p, Eigen::VectorXd params);
