
#include "yao.h"

#include <obliv.h>
#include <stdio.h>
#include <memory.h>

int main(int argc, char* argv[]) {
  ProtocolDesc pd;

  if (argc < 5) {
    fprintf(stderr, "Usage: %s <port> <--|remote host> <config file> <data file> (<validation file>)\n", argv[0]);
    return 1;
  }

  configuration_t* config = GetConfiguration(argv[3]);
  party_t* party = GetParty(config, argv[4], 1);
  int num_features = GetDataFeatureCount(party);
  int num_entries = GetDataRowCount(party);

  long** features = calloc(num_entries, sizeof(long*));
  long* feature_data = calloc(num_entries * num_features, sizeof(long));
  for (int i = 0; i < num_entries; i++) {
    features[i] = feature_data + i * num_features;
  }
  long* labels = calloc(num_entries, sizeof(long));
  QuantizePartyData(party, features, labels, PRECISION);

  connectTcpOrDie(&pd, argv[2], argv[1]);
  fullProtocolIO io = {features, labels, config, party, NULL};
  execYaoProtocol(&pd, do_full_train, &io);
  cleanupProtocol(&pd);

  model_t* model = UnquantizeLongModel(config, io.model, PRECISION);
  double accuracy = EvaluateModel(party, model);
  printf("Training Accuracy: %g\n", accuracy);

  if (strcmp(argv[2], "--") == 0) {
    party_t* validation = GetParty(config, argv[5], 0);
    printf("Validation Accuracy: %g\n", EvaluateModel(validation, model));
  }

  return 0;
}
