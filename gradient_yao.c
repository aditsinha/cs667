
#include "yao.h"

#include <obliv.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <time.h>

#include "wrapper.h"

int main(int argc, char* argv[]) {
  ProtocolDesc pd;

  if (argc < 5) {
    fprintf(stderr, "Usage: %s <port> <--|remote host> <config file> <data file> (<validation file>)\n", argv[0]);
    return 1;
  }

  srand(time(NULL));

  configuration_t* config = GetConfiguration(argv[3]);
  party_t* party = GetParty(config, argv[4], 1, 0);
  model_t* model = InitialModel(config);

  connectTcpOrDie(&pd, argv[2], argv[1]);

  gradientProtocolIO io = {config, party, model};
  
  execYaoProtocol(&pd, do_gradient_train, &io);

  fprintf(stderr, "Completed Yao\n");

  cleanupProtocol(&pd);

  printf("Training Accuracy: %g\n", EvaluateModel(party, model));
  if (strcmp(argv[2], "--") == 0) {
    party_t* validation;
    if (argc == 5) {
      validation = GetParty(config, argv[4], 0, GetDataRowCount(party));
    } else {
      validation = GetParty(config, argv[5], 0, 0);
    }
    printf("Validation Accuracy: %g\n", EvaluateModel(validation, model));
  }

  return 0;
}
