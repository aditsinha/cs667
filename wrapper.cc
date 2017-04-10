
#include <fstream>

#include "party.h"
#include "wrapper.h"

extern "C" configuration_t* GetConfiguration(char* filename);
extern "C" party_t* GetParty(configuration_t* config, char* data_filename);
extern "C" vector_t* ComputeGradient(party_t* party, vector_t* params);
extern "C" vector_t* ComputeNoisyGradient(party_t* party, vector_t* params);

configuration_t* GetConfiguration(char* filename) {
  std::ifstream config_file(filename);

  auto c = new Configuration(config_file);
  auto ct = new configuration_t;
  ct->data = c;
  return ct;
}

party_t* GetParty(configuration_t* config, char* data_filename) {
  std::ifstream data_stream(data_filename);

  auto p = new Party((Configuration*)config->data, data_stream);
  auto pt = new party_t;
  pt->data = p;
  return pt;
}

model_t* InitialModel(configuration_t* config) {
  auto c = (Configuration*)config;

  auto p = new Eigen::VectorXd(c->d);
  auto m = new model_t;
  m->params = p;
  m->config = config;
  
  return m;
}

vector_t* quantizeVector(Eigen::VectorXd vec, int fractional_bits) {
  int shift = 1 << fractional_bits;

  auto v = new vector_t;
  v->length = vec.size();
  v->data = new int[v->length];

  for (int i = 0; i < v->length; i++) {
    v->data[i] = (int)(vec(i) * shift);
  }

  return v;
}

Eigen::VectorXd unquantizeVector(vector_t* vec, int fractional_bits) {
  int shift = 1 << fractional_bits;
  Eigen::VectorXd v(vec->length);

  for (int i = 0; i < vec->length; i++) {
    v(i) = ((double)vec->data[i]) / shift;
  }

  return v;
}

vector_t* ComputeGradient(party_t* party, model_t* model) {
  auto p = (Party*)party->data;
  auto c = (Configuration*)model->config;
  auto dgrad = p->ComputeGradient(c, *(Eigen::VectorXd*)model->params);

  return quantizeVector(dgrad, c->fractional_bits);
}

void UpdateModel(model_t* model, vector_t* gradient) {
  auto c = (Configuration*)model->config;
  auto dgrad = unquantizeVector(gradient, c->fractional_bits);
}

vector_t* ComputeNoisyGradient(party_t* party, model_t* model) {
  auto p = (Party*)party->data;
  auto c = (Configuration*)model->config;
  auto dgrad = p->ComputeGradient(c, *(Eigen::VectorXd*)model->params);

  // figure out and sample noise distribution
}
