
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
    fprintf(stderr, "Usage: %s <port> <--|remote host> <config file> <data file>\n", argv[0]);
    return 1;
  }

  srand(time(NULL));

  configuration_t* config = GetConfiguration(argv[3]);
  party_t* party = GetParty(config, argv[4]);
  model_t* model = InitialModel(config);

  connectTcpOrDie(&pd, argv[2], argv[1]);

  gradientProtocolIO io = {config, party, model};
  
  printf("hello\n");
  execYaoProtocol(&pd, do_gradient_train, &io);

  cleanupProtocol(&pd);

  double accuracy = EvaluateModel(party, model);
  printf("Model Accuracy: %g\n", accuracy);

  return 0;
}
