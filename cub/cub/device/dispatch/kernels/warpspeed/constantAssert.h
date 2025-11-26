/***************************************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cassert> // assert

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
