
#include "obliv_math_def.h"

double sigmoid_taylor_d[TAYLOR_ORDER+1] = {1.0/2, 1.0/4, 0, -1.0/48, 0, 1.0/480, 0, -17.0/80640, 0, 31.0/1451520};
int sigmoid_taylor[TAYLOR_ORDER+1];

void load_sigmoid_taylor_coefficients() {
  int shift = (1 << PRECISION);
  for (int i = 0; i <= TAYLOR_ORDER; i++) {
    sigmoid_taylor[i] = sigmoid_taylor_d[i] * shift;
  }
}
