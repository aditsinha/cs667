
#include "yao.h"

#include <obliv.h>
#include <stdio.h>
#include <memory.h>

int main(int argc, char* argv[]) {
  ProtocolDesc pd;

  if (argc < 5) {
    fprintf(stderr, "Usage: %s <port> <--|remote host> <config file> <data file>\n", argv[0]);
    return 1;
  }

  configuration_t* config = GetConfiguration(argv[3]);
  party_t* party = GetParty(config, argv[4]);
  int num_features = GetDataFeatureCount(party);
  int num_entries = GetDataRowCount(party);

  int** features = calloc(num_entries, sizeof(int*));
  int* feature_data = calloc(num_entries * num_features, sizeof(int));
  for (int i = 0; i < num_entries; i++) {
    features[i] = feature_data + i * num_features;
  }
  int* labels = calloc(num_entries, sizeof(int));
  QuantizePartyData(party, features, labels, PRECISION);

  /* printf("Data\n"); */
  /* for (int i = 0; i < num_features; i++) { */
  /*   printf("%d\n", features[0][i]); */
  /* } */

  connectTcpOrDie(&pd, argv[2], argv[1]);
  fullProtocolIO io = {features, labels, config, party, NULL};
  execYaoProtocol(&pd, do_full_train, &io);
  cleanupProtocol(&pd);

  model_t* model = UnquantizeModel(config, io.model, PRECISION);
  double accuracy = EvaluateModel(party, model);
  printf("Model Accuracy: %g\n", accuracy);

  return 0;
}
