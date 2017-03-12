

#include <eigen3/Eigen/Dense>

#include "party.h"

Eigen::VectorXd train_single(Party* p) {
  Eigen::VectorXd params = Eigen::VectorXd::Zero(p.d);

  learning_rate = .01;

  for (int i = 0; i < 1000; i++) {
    params = params - learning_rate * p->ComputeGradient(params);
  }

  return params;
}
