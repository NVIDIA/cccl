// SPDX-FileCopyrightText: Copyright (c) 2008-2026, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// When compiling with nvcc + nvc++ on amd64, the compilation fails due to nvcc not supporting some AVX512 F16 builtins
// that are used in <immintrin.h>. We can workaround this by pre-including only the relevant mmintrin headers and
// suppressing <immintrin.h> inclusion by TBB.
#if _CCCL_ARCH(X86_64) && _CCCL_COMPILER(NVHPC)
#  include <mmintrin.h>
#  include <xmmintrin.h>

#  define _IMMINTRIN_H_INCLUDED
#  include <oneapi/tbb/detail/_machine.h>
#  undef _IMMINTRIN_H_INCLUDED
#endif // _CCCL_ARCH(X86_64) && _CCCL_COMPILER(NVHPC)
