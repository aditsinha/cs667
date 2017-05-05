
#ifndef YAO_H__
#define YAO_H__

#include "wrapper.h"
#include "obliv.h"

#include "obliv_math_def.h"

typedef struct {
  configuration_t* config;
  party_t* party;
  model_t* model;
} gradientProtocolIO;

void do_gradient_train(void* args);

typedef struct {
  long** features;
  long* labels;
  configuration_t* config;
  party_t* party;
  
  long* model;
} fullProtocolIO;

void do_full_train(void* args);

int connectTcpOrDie(ProtocolDesc* pd, const char* remote, const char* port);

#endif
