// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/detail/__config>

#include <nv/target>

#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

// This file implements Catch2's test macros that work both in host and device code. We globally define the
// CATCH_CONFIG_PREFIX_ALL macro to force Catch2 to prepend it's macros with CATCH_ prefix. That allows us to implement
// the non-prefixed versions ourselves.
//
// In host code, we just use the CATCH_-prefixed variant, in device code we implement the functionality, so it
// corresponds the desired functionality.
//
// Only a subset of the Catch2's macro are provided. If needed, feel free to extend the support. Host-only macros can
// be determined by missing NV_IF_ELSE_TARGET wrapper and immediate dispatch to CATCH_-prefixed variant.

// workaround for error #3185-D: no '#pragma diagnostic push' was found to match this 'diagnostic pop'
#if _CCCL_COMPILER(NVHPC)
#  undef CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION _Pragma("diag push")
#  define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  _Pragma("diag pop")
#endif
// The nv_diagnostic pragmas in Catch2 macros cause cicc to hang indefinitely in CTK 13.0.
// See NVBugs 5475335.
#if _CCCL_VERSION_COMPARE(_CCCL_CTK_, _CCCL_CTK, ==, 13, 0)
#  undef CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#endif
// workaround for error
// * MSVC14.39: #3185-D: no '#pragma diagnostic push' was found to match this 'diagnostic pop'
// * MSVC14.29: internal error: assertion failed: alloc_copy_of_pending_pragma: copied pragma has source sequence entry
//              (pragma.c, line 526 in alloc_copy_of_pending_pragma)
// see also upstream Catch2 issue: https://github.com/catchorg/Catch2/issues/2636
#if _CCCL_COMPILER(MSVC)
#  undef CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS
#  define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS
#endif

// We must pass the COND as a cstring parameter, because it might contain the '%' character that would break the printf
// formatting.
#define C2H_INTERNAL_DEVICE_TEST_PRINT(KIND, COND)                                                               \
  ::printf(                                                                                                      \
    __FILE__                                                                                                     \
    ":" _CCCL_TO_STRING(__LINE__) ":\n    " KIND "(%s) failed\n    block [%u, %u, %u], thread [%u, %u, %u]\n\n", \
    COND,                                                                                                        \
    blockIdx.x,                                                                                                  \
    blockIdx.y,                                                                                                  \
    blockIdx.z,                                                                                                  \
    threadIdx.x,                                                                                                 \
    threadIdx.y,                                                                                                 \
    threadIdx.z)

// <catch2/catch2_test_macros.hpp>

#define REQUIRE(...)                                                             \
  NV_IF_ELSE_TARGET(NV_IS_HOST, (CATCH_REQUIRE(__VA_ARGS__);), ({                \
                      if (!(__VA_ARGS__))                                        \
                      {                                                          \
                        C2H_INTERNAL_DEVICE_TEST_PRINT("REQUIRE", #__VA_ARGS__); \
                        ::__trap();                                              \
                      }                                                          \
                    }))
#define REQUIRE_FALSE(...)                                                             \
  NV_IF_ELSE_TARGET(NV_IS_HOST, (CATCH_REQUIRE_FALSE(__VA_ARGS__);), ({                \
                      if (__VA_ARGS__)                                                 \
                      {                                                                \
                        C2H_INTERNAL_DEVICE_TEST_PRINT("REQUIRE_FALSE", #__VA_ARGS__); \
                        ::__trap();                                                    \
                      }                                                                \
                    }))

#define REQUIRE_THROWS(...)    CATCH_REQUIRE_THROWS(__VA_ARGS__)
#define REQUIRE_THROWS_AS(...) CATCH_REQUIRE_THROWS_AS(__VA_ARGS__)
#define REQUIRE_NOTHROW(...)   NV_IF_ELSE_TARGET(NV_IS_HOST, (CATCH_REQUIRE_NOTHROW(__VA_ARGS__);), (__VA_ARGS__;))

#define CHECK(...)                                                             \
  NV_IF_ELSE_TARGET(NV_IS_HOST, (CATCH_CHECK(__VA_ARGS__);), ({                \
                      if (!(__VA_ARGS__))                                      \
                      {                                                        \
                        C2H_INTERNAL_DEVICE_TEST_PRINT("CHECK", #__VA_ARGS__); \
                        ::__trap();                                            \
                      }                                                        \
                    }))
#define CHECK_FALSE(...)                                                             \
  NV_IF_ELSE_TARGET(NV_IS_HOST, (CATCH_CHECK_FALSE(__VA_ARGS__);), ({                \
                      if (__VA_ARGS__)                                               \
                      {                                                              \
                        C2H_INTERNAL_DEVICE_TEST_PRINT("CHECK_FALSE", #__VA_ARGS__); \
                        ::__trap();                                                  \
                      }                                                              \
                    }))
#define CHECKED_IF(...)   CATCH_CHECKED_IF(__VA_ARGS__)
#define CHECKED_ELSE(...) CATCH_CHECKED_ELSE(__VA_ARGS__)
#define CHECK_NOFAIL(...) CATCH_CHECK_NOFAIL(__VA_ARGS__)

#define CHECK_THROWS(...)    CATCH_CHECK_THROWS(__VA_ARGS__)
#define CHECK_THROWS_AS(...) CATCH_CHECK_THROWS_AS(__VA_ARGS__)
#define CHECK_NOTHROW(...)   NV_IF_ELSE_TARGET(NV_IS_HOST, (CATCH_CHECK_NOTHROW(__VA_ARGS__);), (__VA_ARGS__;))

#define TEST_CASE(...)           CATCH_TEST_CASE(__VA_ARGS__)
#define TEST_CASE_METHOD(...)    CATCH_TEST_CASE_METHOD(__VA_ARGS__)
#define METHOD_AS_TEST_CASE(...) CATCH_METHOD_AS_TEST_CASE(__VA_ARGS__)
#define REGISTER_TEST_CASE(...)  CATCH_REGISTER_TEST_CASE(__VA_ARGS__)
#define SECTION(...)             CATCH_SECTION(__VA_ARGS__)
#define DYNAMIC_SECTION(...)     CATCH_DYNAMIC_SECTION(__VA_ARGS__)
#define FAIL(...)                                                           \
  NV_IF_ELSE_TARGET(NV_IS_HOST, (CATCH_FAIL(__VA_ARGS__);), ({              \
                      C2H_INTERNAL_DEVICE_TEST_PRINT("FAIL", #__VA_ARGS__); \
                      ::__trap();                                           \
                    }))
#define FAIL_CHECK(...) CATCH_FAIL_CHECK(__VA_ARGS__)
#define SUCCEED(...)    CATCH_SUCCEED(__VA_ARGS__)
#define SKIP(...)       CATCH_SKIP(__VA_ARGS__)

#define STATIC_REQUIRE(...) \
  NV_IF_ELSE_TARGET(NV_IS_HOST, (CATCH_STATIC_REQUIRE(__VA_ARGS__);), (static_assert(__VA_ARGS__, #__VA_ARGS__);))
#define STATIC_REQUIRE_FALSE(...) \
  NV_IF_ELSE_TARGET(              \
    NV_IS_HOST, (CATCH_STATIC_REQUIRE_FALSE(__VA_ARGS__);), (static_assert(!(__VA_ARGS__), "!(" #__VA_ARGS__ ")");))
#define STATIC_CHECK(...) \
  NV_IF_ELSE_TARGET(NV_IS_HOST, (CATCH_STATIC_CHECK(__VA_ARGS__);), (static_assert(__VA_ARGS__, #__VA_ARGS__);))
#define STATIC_CHECK_FALSE(...) \
  NV_IF_ELSE_TARGET(            \
    NV_IS_HOST, (CATCH_STATIC_CHECK_FALSE(__VA_ARGS__);), (static_assert(!(__VA_ARGS__), "!(" #__VA_ARGS__ ")");))

#define SCENARIO(...)        CATCH_SCENARIO(__VA_ARGS__)
#define SCENARIO_METHOD(...) CATCH_SCENARIO_METHOD(__VA_ARGS__)
#define GIVEN(...)           CATCH_GIVEN(__VA_ARGS__)
#define AND_GIVEN(...)       CATCH_AND_GIVEN(__VA_ARGS__)
#define WHEN(...)            CATCH_WHEN(__VA_ARGS__)
#define AND_WHEN(...)        CATCH_AND_WHEN(__VA_ARGS__)
#define THEN(...)            CATCH_THEN(__VA_ARGS__)
#define AND_THEN(...)        CATCH_AND_THEN(__VA_ARGS__)

// <catch2/catch_message.hpp>

#define INFO(...)          CATCH_INFO(__VA_ARGS__)
#define UNSCOPED_INFO(...) CATCH_UNSCOPED_INFO(__VA_ARGS__)
#define WARN(...)          CATCH_WARN(__VA_ARGS__)
#define CAPTURE(...)       CATCH_CAPTURE(__VA_ARGS__)

// <catch2/catch_template_test_macros.hpp>

#define TEMPLATE_TEST_CASE(...)                    CATCH_TEMPLATE_TEST_CASE(__VA_ARGS__)
#define TEMPLATE_TEST_CASE_SIG(...)                CATCH_TEMPLATE_TEST_CASE_SIG(__VA_ARGS__)
#define TEMPLATE_TEST_CASE_METHOD(...)             CATCH_TEMPLATE_TEST_CASE_METHOD(__VA_ARGS__)
#define TEMPLATE_TEST_CASE_METHOD_SIG(...)         CATCH_TEMPLATE_TEST_CASE_METHOD_SIG(__VA_ARGS__)
#define TEMPLATE_PRODUCT_TEST_CASE(...)            CATCH_TEMPLATE_PRODUCT_TEST_CASE(__VA_ARGS__)
#define TEMPLATE_PRODUCT_TEST_CASE_SIG(...)        CATCH_TEMPLATE_PRODUCT_TEST_CASE_SIG(__VA_ARGS__)
#define TEMPLATE_PRODUCT_TEST_CASE_METHOD(...)     CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD(__VA_ARGS__)
#define TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG(...) CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG(__VA_ARGS__)
#define TEMPLATE_LIST_TEST_CASE(...)               CATCH_TEMPLATE_LIST_TEST_CASE(__VA_ARGS__)
#define TEMPLATE_LIST_TEST_CASE_METHOD(...)        CATCH_TEMPLATE_LIST_TEST_CASE_METHOD(__VA_ARGS__)

// <catch2/matchers/catch_matchers.hpp>

#define REQUIRE_THROWS_WITH(...)    CATCH_REQUIRE_THROWS_WITH(__VA_ARGS__)
#define REQUIRE_THROWS_MATCHES(...) CATCH_REQUIRE_THROWS_MATCHES(__VA_ARGS__)
#define CHECK_THROWS_WITH(...)      CATCH_CHECK_THROWS_WITH(__VA_ARGS__)
#define CHECK_THROWS_MATCHES(...)   CATCH_CHECK_THROWS_MATCHES(__VA_ARGS__)
#define CHECK_THAT(...)             CATCH_CHECK_THAT(__VA_ARGS__)
#define REQUIRE_THAT(...)           CATCH_REQUIRE_THAT(__VA_ARGS__)

// extensions

// Sometimes clang-cuda has problems with REQUIRE(...) when used in __device__ function - it tries to instantiate the
// host path. This is related to clang-cuda's compilation trajectory. For these cases, we provide REQUIRE_DEVICE(...) as
// a fallback.
#define REQUIRE_DEVICE(...)                                    \
  do                                                           \
  {                                                            \
    if (!(__VA_ARGS__))                                        \
    {                                                          \
      C2H_INTERNAL_DEVICE_TEST_PRINT("REQUIRE", #__VA_ARGS__); \
      ::__trap();                                              \
    }                                                          \
  } while (false)

// Macros to require/check success of a CUDA Driver call.
#define REQUIRE_CUDA(...) REQUIRE((__VA_ARGS__) == CUDA_SUCCESS)
#define CHECK_CUDA(...)   CHECK((__VA_ARGS__) == CUDA_SUCCESS)

// Macros to require/check success of a CUDA Runtime call.
#define REQUIRE_CUDART(...) REQUIRE((__VA_ARGS__) == cudaSuccess)
#define CHECK_CUDART(...)   CHECK((__VA_ARGS__) == cudaSuccess)
