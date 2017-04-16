
#include <fstream>
#include <iostream>

#include "party.h"
#include "privacy.h"
#include "wrapper.h"
  
configuration_t* GetConfiguration(char* filename) {
  std::ifstream config_file(filename);

  auto c = new Configuration(config_file);

  return new configuration_t{c};
}

party_t* GetParty(configuration_t* config, char* data_filename) {
  std::ifstream data_stream(data_filename);

  auto p = new Party((Configuration*)config->data, data_stream);
  
  return new party_t{p};
}

model_t* InitialModel(configuration_t* config) {
  auto c = (Configuration*)config->data;

  auto p = new Eigen::VectorXd(c->d);
  p->setZero();
  return new model_t{p, c};
}

model_t* UnquantizeModel(configuration_t* config, int* quantized, int precision) {
  auto c = (Configuration*)config->data;
  model_t* m = InitialModel(config);
  Eigen::VectorXd* params = (Eigen::VectorXd*)m->params;
  for (int i = 0; i < c->d; i++) {
    (*params)(i) = quantized[i];
  }

  (*params) /= (1 << precision);

  return m;
}

int* quantizeVector(Eigen::VectorXd vec, int fractional_bits) {
  int shift = 1 << fractional_bits;

  auto v = new int[vec.size()];

  for (int i = 0; i < vec.size(); i++) {
    v[i] = (int)(vec(i) * shift);
  }

  return v;
}

Eigen::VectorXd unquantizeVector(int* vec, int length, int fractional_bits) {
  int shift = 1 << fractional_bits;
  Eigen::VectorXd v(length);

  for (int i = 0; i < length; i++) {
    v(i) = ((double)vec[i]) / shift;
  }

  return v;
}

int* ComputeGradient(party_t* party, model_t* model) {
  auto p = (Party*)party->data;
  auto c = (Configuration*)model->config;

  auto dgrad = p->ComputeGradient(c, *(Eigen::VectorXd*)model->params);

  return quantizeVector(dgrad, c->fractional_bits);
}

void UpdateModel(model_t* model, int step_num, int* gradient) {
  auto c = (Configuration*)model->config;
  auto dgrad = unquantizeVector(gradient, c->d, c->fractional_bits);

  int batches_per_epoch = c->m / c->batch_size;
  int epoch_num = step_num / batches_per_epoch;
  double learning_rate = c->initial_learning_rate / (1 + c->learning_rate_decay * epoch_num);

  dgrad /= c->n;

  Eigen::VectorXd* params = (Eigen::VectorXd*)model->params;
  (*params) -= learning_rate * dgrad;
}

int* ComputeNoisyGradient(party_t* party, model_t* model) {
  auto p = (Party*)party->data;
  auto c = (Configuration*)model->config;
  auto dgrad = p->ComputeGradient(c, *(Eigen::VectorXd*)model->params);

  // sample noise distribution
  dgrad += c->privacy.generateLogisticRegressionNoise(c->clipping, c->batch_size, c->m, c->epochs, c->d);
  return quantizeVector(dgrad, c->fractional_bits);
}

int GetIterationCount(configuration_t* config) {
  auto c = (Configuration*)config->data;
  return c->epochs * c->m / c->batch_size;
}

int GetNumFeatures(configuration_t* config) {
  return ((Configuration*)config->data)->d;
}

double EvaluateModel(party_t* party, model_t* model) {
  auto p = (Party*)party->data;

  return p->Accuracy(*(Eigen::VectorXd*)model->params);
}

int GetDataFeatureCount(party_t* party) {
  auto p = (Party*)party->data;
  return p->features.cols();
}

int GetDataRowCount(party_t* party) {
  auto p = (Party*)party->data;
  return p->features.rows();
}

int GetBatchSize(configuration_t* config) {
  auto c = (Configuration*)config->data;
  return c->batch_size;
}

int GetQuantizeBitsPrecision(configuration_t* config) {
  auto c = (Configuration*)config->data;
  return c->fractional_bits;
}

// assume that features and labels have already been allocated and are of the correct size
void QuantizePartyData(party_t* party, int** features, int* labels, int precision) {
  auto p = (Party*)party->data;

  Eigen::MatrixXi int_mat = (p->features * (1 << precision)).cast<int>();

  for (int i = 0; i < int_mat.rows(); i++) {
    for (int j = 0; j < int_mat.cols(); j++) {
      features[i][j] = int_mat(i,j);
    }

    labels[i] = (int)p->labels(i);
  }
}

double GetLearningRate(configuration_t* config, int iteration) {
  auto c = (Configuration*)config->data;

  // we need to include the number of parties because the batches in
  // the full yao implementation are over the combined datasets.
  int batches_per_epoch = c->n * c->m / c->batch_size;
  int epoch_num = iteration / batches_per_epoch;
  return c->initial_learning_rate / (1 + c->learning_rate_decay * epoch_num);
}
