
#ifndef OBLIV_MATH_FUNC_H__
#define OBLIV_MATH_FUNC_H__

#include "obliv_math_def.h"

#ifdef USE_OBLIV_INT
#include <obliv.oh>
typedef obliv long oint;

#define DEBUG(...)
#define OVERFLOW_DETECT_MULT(x,y,z)
#define OVERFLOW_DETECT_ADD(x,y,z)

#else
typedef long oint;
#include <stdio.h>
#include <assert.h>
#define DEBUG(...) fprintf (stderr, __VA_ARGS__)

#define OVERFLOW_DETECT_MULT(x,y,z)	\
  if (x != 0 && z / x != y) {		\
    fprintf(stderr, "Mult Overflow\n"); \
    assert(0);				\
  }

#define OVERFLOW_DETECT_ADD(x,y,z)     \
  if ((x > 0 && y > 0 && z < x) ||     \
      (x < 0 && y < 0 && z > x)) {     \
    fprintf(stderr, "Add Overflow\n"); \
    assert(0);			       \
  }
  
#endif

static inline oint add_oo(oint x, oint y) {
  oint z = x+y;
  OVERFLOW_DETECT_ADD(x,y,z);
  return z;
}

static inline oint mult_oo(oint x, oint y) {
  oint z = x*y; 
  OVERFLOW_DETECT_MULT(x,y,z);
  return z >> PRECISION;
}

static inline oint mult_op(oint x, long y) {
  oint z = x*y;
  OVERFLOW_DETECT_MULT(x,y,z);
  return z >> PRECISION;
}

static inline void mult_ovec_p(oint* vec, long p, int num_entries) {
  for (int i = 0; i < num_entries; i++) {
    vec[i] = mult_op(vec[i], p);
  }
}

static inline void add_ovecs(oint* vec1, oint* vec2, int num_entries) {
  for (int i = 0; i < num_entries; i++) {
    oint v = vec1[i] + vec2[i];
    OVERFLOW_DETECT_ADD(vec1[i], vec2[i], v);
    vec1[i] = v;
  }
}

static inline oint evalPolynomial(oint x, long* coeffs, int degree) {
  // assume that the first coefficient in the constant term, the second is the linear term etc.
  oint val = coeffs[degree];

  for (int i = degree-1; i >= 0; i--) {
    val = mult_oo(val, x);
    val = add_oo(val, coeffs[i]);
  }

  return val;
}

// use a taylor series to approximate the result
static inline oint oblivious_logistic_fn(oint* model, oint* features, int num_features) {
  oint dot_product = 0;
  for (int i = 0; i < num_features; i++) {
    dot_product = add_oo(dot_product, mult_oo(model[i+1], features[i]));
  }

  dot_product = add_oo(dot_product, model[0]); // bias term

  return evalPolynomial(dot_product, sigmoid_taylor, TAYLOR_ORDER);
}

static inline void add_to_gradient(oint* grad, oint* model, oint* features, oint label, int num_features) {
  oint err = oblivious_logistic_fn(model, features, num_features) - (label << PRECISION);

  grad[0] = add_oo(grad[0], err);
  for (int k = 0; k < num_features; k++) {
    grad[k+1] = add_oo(grad[k+1], mult_oo(err, features[k]));
  }
}

static inline void add_regularization(oint* grad, oint* model, long reg_factor, int num_features) {
  // don't regularize bias term
  for (int k = 1; k < num_features+1; k++) {
    grad[k] = add_oo(grad[k], mult_op(model[k], reg_factor));
  }
}

#endif
