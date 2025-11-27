// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cassert> // assert

/*
 * constantAssert: an assertion that is intended to be verified at compile time.
 *
 * A constantAssert asserts something that the compiler (optimizer) can verify
 * at compile time. Therefore, it does not result in an actual call to assert in
 * the compiled binary. This allows checking various properties that cannot be
 * verified using static_assert.
 *
 * To ensure that all constantAsserts are in fact eliminated, compile with
 * -DWARPSPEED_FORCE_ASSERT_AT_COMPILE_TIME. With this macro defined, any
 * constantAssert failure will output illegal PTX containing the error message.
 * As a result, compilation will fail.
 *
 * Compiling with -DWARPSPEED_FORCE_ASSERT_AT_COMPILE_TIME has the additional
 * advantage that violating any of the assertions can be detected at compile
 * time and before even running the code.
 *
 */

#define WS_STRINGIZE_DETAIL(x) #x
#define WS_STRINGIZE(x)        WS_STRINGIZE_DETAIL(x)

#if defined(WARPSPEED_FORCE_ASSERT_AT_COMPILE_TIME) && defined(__CUDA_ARCH__)
// When WARPSPEED_FORCE_ASSERT_AT_COMPILE_TIME is defined and compiling for device, output illegal PTX.
// This causes the compilation to fail.
#  define constantAssert(expr, msg)                                                            \
    do                                                                                         \
    {                                                                                          \
      if (!(expr))                                                                             \
      {                                                                                        \
        asm volatile(".pragma \"\n" __FILE__ "(" WS_STRINGIZE(                                 \
          __LINE__) "): %0"                                                                    \
                    ": error: constant assertion failed with '" msg "'\n\";" ::"C"(__func__)); \
      }                                                                                        \
    } while (0)
#else
// Host or !WARPSPEED_FORCE_ASSERT_AT_COMPILE_TIME
#  define constantAssert(expr, msg) (assert((expr) && (msg)))
#endif
