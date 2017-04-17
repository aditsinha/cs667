
#include "obliv_math_def.h"

#include <stdlib.h>

double sigmoid_taylor_d[] = {1.0/2, 1.0/4, 0, -1.0/48, 0, 1.0/480, 0, -17.0/80640, 0, 31.0/1451520};
long* sigmoid_taylor;

void load_sigmoid_taylor_coefficients() {
  sigmoid_taylor = malloc((TAYLOR_ORDER + 1)*sizeof(long));
  long shift = (1L << PRECISION);
  for (int i = 0; i <= TAYLOR_ORDER; i++) {
    sigmoid_taylor[i] = sigmoid_taylor_d[i] * shift;
  }
}
