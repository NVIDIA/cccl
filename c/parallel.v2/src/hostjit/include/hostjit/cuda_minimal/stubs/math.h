#ifndef _HOSTJIT_MATH_H
#define _HOSTJIT_MATH_H

// Macros needed by __clang_cuda_math.h
#define HUGE_VAL         __builtin_huge_val()
#define HUGE_VALF        __builtin_huge_valf()
#define HUGE_VALL        __builtin_huge_vall()
#define INFINITY         __builtin_inff()
#define NAN              __builtin_nanf("")
#define MATH_ERRNO       1
#define MATH_ERREXCEPT   2
#define math_errhandling (MATH_ERRNO | MATH_ERREXCEPT)
#define FP_NAN           0
#define FP_INFINITE      1
#define FP_ZERO          2
#define FP_SUBNORMAL     3
#define FP_NORMAL        4
#define __signbit(x)     __builtin_signbit(x)
#define __signbitl(x)    __builtin_signbitl(x)

#endif
