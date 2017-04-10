
#ifndef WRAPPER_H_
#define WRAPPER_H_

typedef struct {
  void* data;
} configuration_t;

typedef struct {
  void* data
} party_t;

typedef struct {
  int* data;
  int length;
} vector_t;

typedef struct {
  void* params;
  void* config;
} model_t;

#endif
