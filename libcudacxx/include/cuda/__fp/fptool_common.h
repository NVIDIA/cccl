//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPTOOL_COMMON_H
#define _CUDA___FP_FPTOOL_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fptool_common.h - The opt-in gate shared by the FPTOOL headers
    ======================================================================================================
    Unlike <cuda/fpmp> and <cuda/fpemu>, which are arithmetic types and nothing more,
    fptool is a diagnostic instrument, and both of its types carry mutable state at
    namespace scope:

      - fp_custom's runtime mantissa and exponent sizes, one host copy and one device
        copy per component type (<cuda/__fp/fptool_custom.h>)
      - fpmp2_stat's operation counters, which every instrumented operation updates with
        an atomicAdd (<cuda/__fp/fptool_stat.h>)

    Those are variable templates, so they have vague linkage and one copy is shared by
    every translation unit that includes the header. That is what the feature needs in
    order to work, and it is also why merely including the header is not free: an
    #include left behind after an experiment puts the state, and the counter traffic on
    every fpmp2_stat operation, into a shipping binary.

    So the feature is opt-in. CCCL_ENABLE_FPTOOL has to be defined for the whole project,
    the way the other CCCL configuration macros are, before any CCCL header is included:

        nvcc -DCCCL_ENABLE_FPTOOL ...

    Defining it for only some translation units is the one thing to avoid, since the
    state above is shared across all of them.
*/

// The header-test targets compile every public and private header as a bare #include
// with no project configuration, so they cannot define the macro. Letting them through
// keeps fptool covered by that test rather than excluded from it; _CCCL_HEADER_TEST is
// internal to CCCL's build and is not something a user program defines.
#if !defined(CCCL_ENABLE_FPTOOL) && !defined(_CCCL_HEADER_TEST)
#  error \
    "<cuda/fptool> is opt-in, because fp_custom and fpmp2_stat carry mutable state at namespace scope that is shared across translation units. Define CCCL_ENABLE_FPTOOL for the whole project to enable it."
#endif // !CCCL_ENABLE_FPTOOL && !_CCCL_HEADER_TEST

#endif // _CUDA___FP_FPTOOL_COMMON_H
