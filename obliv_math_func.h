
#ifndef OBLIV_MATH_FUNC_H__
#define OBLIV_MATH_FUNC_H__

#include "obliv_math_def.h"

#ifdef USE_OBLIV_INT
#include <obliv.oh>
typedef obliv int oint;

#define DEBUG(...)
#else
typedef long oint;
#include <stdio.h>
#define DEBUG(...) fprintf (stderr, __VA_ARGS__)

#endif

static inline oint mult_oo(oint x, oint y) {
  return (x*y) / (1L << PRECISION);
}

static inline oint mult_op(oint x, long y) {
  return (x*y) / (1L << PRECISION);
}

static inline void mult_ovec_p(oint* vec, long p, int num_entries) {
  for (int i = 0; i < num_entries; i++) {
    vec[i] = mult_op(vec[i], p);
  }
}

static inline void add_ovecs(oint* vec1, oint* vec2, int num_entries) {
  for (int i = 0; i < num_entries; i++) {
    vec1[i] = vec1[i] + vec2[i];
  }
}

static inline oint evalPolynomial(oint x, long* coeffs, int degree) {
  // assume that the first coefficient in the constant term, the second is the linear term etc.
  oint val = coeffs[degree];

  for (int i = degree-1; i >= 0; i--) {
    val = mult_oo(val, x);
    val = val + coeffs[i];
  }

  return val;
}

// use a taylor series to approximate the result
static inline oint oblivious_logistic_fn(oint* model, oint* features, int num_features) {
  oint dot_product = 0;
  for (int i = 0; i < num_features; i++) {
    dot_product += mult_oo(model[i], features[i]);
  }

  return evalPolynomial(dot_product, sigmoid_taylor, TAYLOR_ORDER);
}

static inline void add_to_gradient(oint* grad, oint* model, oint* features, oint label, int num_features) {
  oint err = oblivious_logistic_fn(model, features, num_features) - (label << PRECISION);

  for (int k = 0; k < num_features; k++) {
    grad[k] += mult_oo(err, features[k]);
  }
}

#endif
