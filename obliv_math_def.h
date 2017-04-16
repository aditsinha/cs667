
#ifndef OBLIV_MATH_DEF_H__
#define OBLIV_MATH_DEF_H__

#define PRECISION 20
#define TAYLOR_ORDER 9
extern double sigmoid_taylor_d[TAYLOR_ORDER+1];
extern int sigmoid_taylor[TAYLOR_ORDER+1];

#ifdef __cplusplus
extern "C" {
#endif
  void load_sigmoid_taylor_coefficients();
#ifdef __cplusplus
}
#endif


#endif
