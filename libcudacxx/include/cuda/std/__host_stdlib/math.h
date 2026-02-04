//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___HOST_STDLIB_MATH_H
#define _CUDA_STD___HOST_STDLIB_MATH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HOSTED()
#  include <math.h>

// Standard C++ library comes with it's own <math.h> C++ compatible header. However, if the include paths are jumbled,
// it might happen that the original C <math.h> is found first. This is a problem because C headers define many of the
// math functions as macros which would change our definitions. So, we check whether any of the functions are defined
// as a macro to distinguish the C++ copatibility header from the C header.
#  if defined(fabs) || defined(fmod) || defined(remainder) || defined(remquo) || defined(fma) || defined(fmax)      \
    || defined(fmin) || defined(fdim) || defined(exp) || defined(exp2) || defined(expm1) || defined(log)            \
    || defined(log10) || defined(log2) || defined(log1p) || defined(pow) || defined(sqrt) || defined(cbrt)          \
    || defined(hypot) || defined(sin) || defined(cos) || defined(tan) || defined(asin) || defined(acos)             \
    || defined(atan) || defined(atan2) || defined(sinh) || defined(cosh) || defined(tanh) || defined(asinh)         \
    || defined(acosh) || defined(atanh) || defined(erf) || defined(erfc) || defined(tgamma) || defined(lgamma)      \
    || defined(ceil) || defined(floor) || defined(trunc) || defined(round) || defined(lround) || defined(llround)   \
    || defined(nearbyint) || defined(rint) || defined(lrint) || defined(llrint) || defined(frexp) || defined(ldexp) \
    || defined(scalbn) || defined(scalbln) || defined(ilogb) || defined(logb) || defined(nextafter)                 \
    || defined(nexttoward) || defined(copysign) || defined(fpclassify) || defined(isfinite) || defined(isinf)       \
    || defined(isnan) || defined(isnormal) || defined(signbit) || defined(isgreater) || defined(isgreaterequal)     \
    || defined(isless) || defined(islessequal) || defined(islessgreater) || defined(isunordered)
#    error \
      "libcu++ requires the C++ compatibility <math.h> header, not the C <math.h> header. Please, check your include paths."
#  endif // math functions defined as macros

#endif // _CCCL_HOSTED()

#endif // _CUDA_STD___HOST_STDLIB_MATH_H
