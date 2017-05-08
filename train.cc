
#include <eigen3/Eigen/Dense>

#include "party.h"
#include "train.h"
#include <vector>

double getLearningRate(Configuration* c, int iteration) {
  int batches_per_epoch = c->m / c->batch_size;
  int epoch_num = iteration / batches_per_epoch;
  return c->initial_learning_rate / (1 + c->learning_rate_decay * epoch_num);
}

double getLearningRate(double initial, double decay, int epoch) {
  return initial / (1 + decay * epoch);
}

Eigen::VectorXd gradient_train_simulation(std::vector<Party*> p, Configuration* c, bool use_noise) {
  // TODO need to change the way noise is generated
  Eigen::VectorXd params = Eigen::VectorXd::Zero(c->d + 1);

  int num_batches_per_epoch = c->m / c->batch_size;

  int effective_epochs = c->epochs / sqrt(c->n);

  for (int i = 0; i < effective_epochs; i++) {
    double learning_rate = getLearningRate(c, i*num_batches_per_epoch);
    
    std::cout << "Epoch " << i << std::endl;
    
    for (int j = 0; j < num_batches_per_epoch; j++) {
      Eigen::VectorXd gradient = Eigen::VectorXd::Zero(c->d + 1);
    
      for (auto party : p) {
	auto local_gradient = party->ComputeGradient(c, params);
	if (use_noise && c->n > 2) {
	  // and multiparty noise
	  local_gradient +=
	    c->privacy.generateLogisticRegressionMPCNoise(c->clipping,
							  c->batch_size, c->m,
							  effective_epochs, c->d, c->n);
	  local_gradient = reduceVectorPrecision(local_gradient, c->fractional_bits);
	}

	gradient += local_gradient;
      }

      if (use_noise && (c->n == 1 || c-> n == 2)) {
      	// add solo/two party noise
      	gradient +=
      	  reduceVectorPrecision(c->privacy.generateLogisticRegressionNoise(c->clipping,
      									   c->batch_size, c->m,
      									   effective_epochs, c->d),
      				c->fractional_bits);
      }



      params = params - gradient * learning_rate;
    }
  }

  std::cout << "Training Accuracy " << p[0]->Accuracy(params) << std::endl;

  // std::cout << std::endl << params << std::endl;

  return params;
}

Eigen::VectorXd train_single(Party* p, Configuration* c) {
  Eigen::VectorXd params = Eigen::VectorXd::Zero(c->d + 1);

  float learning_rate = c->initial_learning_rate;

  int num_batches_per_epoch = c->m / c->batch_size;

  for (int i = 0; i < c->epochs; i++) {
    std::cout << "Epoch " << i << std::endl;
    
    for (int j = 0; j < num_batches_per_epoch; j++) {
      auto gradient = p->ComputeGradient(c, params);

      params = params - gradient * learning_rate;
    }

    learning_rate = getLearningRate(c, i*num_batches_per_epoch);
  }

  return params;
}
