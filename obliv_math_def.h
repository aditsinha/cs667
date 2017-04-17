
#ifndef OBLIV_MATH_DEF_H__
#define OBLIV_MATH_DEF_H__

#define PRECISION 31
#define TAYLOR_ORDER 9
extern double sigmoid_taylor_d[];
extern long* sigmoid_taylor;

#ifdef __cplusplus
extern "C" {
#endif
  void load_sigmoid_taylor_coefficients();
#ifdef __cplusplus
}
#endif

#endif