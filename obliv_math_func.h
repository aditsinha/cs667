
#ifndef OBLIV_MATH_FUNC_H__
#define OBLIV_MATH_FUNC_H__

#include "obliv_math_def.h"

#include <stdlib.h>
#include <math.h>

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
    fprintf(stderr, "Mult Overflow: %ld x %ld\n", x, y);	\
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

static inline oint generate_binomial_noise(double sd, int prec) {
  // a binomial distribution from an unbiased coin has sd =
  // .25*sqrt(n), where n is the number of trials.  also, for a random
  // variable x with sd(x) = s, sd(x/a) = s/a.
  int bits_needed = ceil((1 << 2*prec) * sd * sd);
  oint result = 0;

  // we can generate the randomness in groups of 64 using long longs
  for (int j = 0; j < bits_needed; j += 64) {
    // rand max is at least 2^15

#ifdef USE_OBLIV_INT    
    unsigned long long r1 = rand(), r2 = rand(), r3 = rand(), r4 = rand(), r5 = rand();
    unsigned long long r = r1 ^ (r2 << 15) ^ (r3 << 30) ^ (r4 << 45) ^ (r5 << 60);

    obliv unsigned long long or1 = feedOblivLLong(r, 1);
    obliv unsigned long long or2 = feedOblivLLong(r, 2);
    obliv unsigned long long rand_bits = or1 ^ or2;

    if (bits_needed < 64) {
      // we want zero out everything except the bits needed
      for (int i = 63; i > bits_needed; i--) {
    	rand_bits &= ~(1ULL << i);
      }
    }

/* #ifdef USE_OBLIV_INT */
/*     long long res; */
/*     revealOblivLLong(&res, rand_bits, 0); */
/*     printf("rand: %ld\n", res); */
/* #endif */
#else
    unsigned long long r1 = rand(), r2 = rand(), r3 = rand(), r4 = rand(), r5 = rand();
    unsigned long long rand_bits = r1 ^ (r2 << 15) ^ (r3 << 30) ^ (r4 << 45) ^ (r5 << 60);
#endif

    // now get the hamming weight
    rand_bits = ((rand_bits & 0xAAAAAAAAAAAAAAAALL) >> 1) + (rand_bits & 0x5555555555555555LL);
    rand_bits = ((rand_bits & 0xCCCCCCCCCCCCCCCCLL) >> 2) + (rand_bits & 0x3333333333333333LL);
    rand_bits = ((rand_bits & 0xF0F0F0F0F0F0F0F0LL) >> 4) + (rand_bits & 0x0F0F0F0F0F0F0F0FLL);
    rand_bits = ((rand_bits & 0xFF00FF00FF00FF00LL) >> 8) + (rand_bits & 0x00FF00FF00FF00FFLL);
    rand_bits = ((rand_bits & 0xFFFF0000FFFF0000LL) >> 16) + (rand_bits & 0x0000FFFF0000FFFFLL);
    rand_bits = ((rand_bits & 0xFFFFFFFF00000000LL) >> 32) + (rand_bits & 0x00000000FFFFFFFFLL);

    // and add that to the result
    result += rand_bits;
  }

  // we want to center the randomness around 0
  result -= (bits_needed / 2);

  return result;
}

static inline oint* generate_noise_vec(int d, double std_dev, int prec) {
  oint* noise = (oint*)calloc(d, sizeof(oint));
  for (int i = 0; i < d; i++) {
    noise[i] = generate_binomial_noise(std_dev, prec);
  }
  return noise;
}


#endif
