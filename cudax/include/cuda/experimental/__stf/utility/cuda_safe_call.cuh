//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @dir utility
 *
 * @brief Utility functions and classes
 */
/**
 * @file
 * @brief Facilities for error detection and error handling
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/source_location>

#include <cuda/experimental/__stf/utility/unittest.cuh>

#include <cuda.h>
#include <cuda_occupancy.h>
#include <cuda_runtime.h>

#if _CCCL_HAS_INCLUDE(<cusolverDn.h>)
#  include <cusolverDn.h>
#endif

namespace cuda::experimental::stf
{

#if _CCCL_HAS_INCLUDE(<cusolverDn.h>)
// Undocumented
inline const char* cusolverGetErrorString(const cusolverStatus_t status)
{
  switch (status)
  {
    default:
      break;
#  define _b738a2a5fe81ee876deadae4a109521c(x) \
    case x:                                    \
      return #x
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_SUCCESS);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_NOT_INITIALIZED);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_ALLOC_FAILED);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_INVALID_VALUE);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_ARCH_MISMATCH);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_MAPPING_ERROR);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_EXECUTION_FAILED);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_INTERNAL_ERROR);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_NOT_SUPPORTED);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_ZERO_PIVOT);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_INVALID_LICENSE);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_IRS_PARAMS_INVALID);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_IRS_INTERNAL_ERROR);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_IRS_NOT_SUPPORTED);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_IRS_OUT_OF_RANGE);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_IRS_MATRIX_SINGULAR);
      _b738a2a5fe81ee876deadae4a109521c(CUSOLVER_STATUS_INVALID_WORKSPACE);
#  undef _b738a2a5fe81ee876deadae4a109521c
  }
  return "Unknown cuSOLVER status";
}
#endif

/**
 * @brief Exception type across CUDA, CUBLAS, and CUSOLVER.
 *
 * @paragraph example Example
 * @snippet this cuda_exception
 */
class cuda_exception : public ::std::exception
{
public:
  cuda_exception() = delete;
  // TODO (miscco): Why was this not copyable?
  // cuda_exception(const cuda_exception&) = delete;

  /**
   * @brief Constructs an exception object from a status value.
   *
   * If `status` is `0`, the exception is still created with an empty error message. Otherwise, the constructor
   * initializes the error message (later accessible with `what()`) appropriately.
   *
   * @tparam T status type, can be `cudaError_t`, `cublasStatus_t`, or `cusolverStatus_t`
   * @param status status value, usually the result of a CUDA API call
   * @param loc location of the call, defaulted
   */
  template <typename T>
  cuda_exception(const T status, const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
  {
    // All "success" statuses are zero
    static_assert(cudaSuccess == 0 && CUDA_SUCCESS == 0
#if _CCCL_HAS_INCLUDE(<cublas_v2.h>)
                    && CUBLAS_STATUS_SUCCESS == 0
#endif
#if _CCCL_HAS_INCLUDE(<cusolverDn.h>)
                    && CUSOLVER_STATUS_SUCCESS == 0
#endif
                  ,
                  "Please revise this function.");

    // Common early exit test for all cases
    if (status == 0)
    {
      return;
    }

    int dev = -1;
    cudaGetDevice(&dev);

#if _CCCL_HAS_INCLUDE(<cusolverDn.h>)
    if constexpr (::std::is_same_v<T, cusolverStatus_t>)
    {
      format("%s(%u) [device %d] CUSOLVER error in call %s: %s.",
             loc.file_name(),
             loc.line(),
             dev,
             loc.function_name(),
             cusolverGetErrorString(status));
    }
    else
#endif // _CCCL_HAS_INCLUDE(<cusolverDn.h>)
#if _CCCL_HAS_INCLUDE(<cublas_v2.h>)
      if constexpr (::std::is_same_v<T, cublasStatus_t>)
    {
      format("%s(%u) [device %d] CUBLAS error in %s: %s.",
             loc.file_name(),
             loc.line(),
             dev,
             loc.function_name(),
             cublasGetStatusString(status));
    }
    else
#endif // _CCCL_HAS_INCLUDE(<cublas_v2.h>)
      if constexpr (::std::is_same_v<T, cudaOccError>)
      {
        format("%s(%u) [device %d] CUDA OCC error in %s: %s.",
               loc.file_name(),
               loc.line(),
               dev,
               loc.function_name(),
               cudaGetErrorString(cudaErrorInvalidConfiguration));
      }
      else if constexpr (::std::is_same_v<T, CUresult>)
      {
        const char* error_string = nullptr;
        cuGetErrorString(status, &error_string);
        const char* error_name = nullptr;
        cuGetErrorName(status, &error_name);
        format("%s(%u) [device %d] CUDA DRIVER error in %s: %s (%s).",
               loc.file_name(),
               loc.line(),
               dev,
               loc.function_name(),
               error_string,
               error_name);
      }
      else
      {
        static_assert(::std::is_same_v<T, cudaError_t>, "Error: not a CUDA status.");
        format("%s(%u) [device %d] CUDA error in %s: %s (%s).",
               loc.file_name(),
               loc.line(),
               dev,
               loc.function_name(),
               cudaGetErrorString(status),
               cudaGetErrorName(status));
      }
  }

  /**
   * @brief Returns a message describing the error.
   *
   * @return the error message
   */
  const char* what() const noexcept override
  {
    return msg.c_str();
  }

private:
  template <typename... Ps>
  void format(const char* fmt, Ps&&... ps)
  {
    // Compute the bytes to be written.
    auto needed = ::std::snprintf(nullptr, 0, fmt, ps...);
    // Pedantically reserve one extra character for the terminating `\0`.
    msg.resize(needed + 1);
    // This will write `needed` bytes plus a `\0` at the end.
    ::std::snprintf(&msg[0], msg.capacity(), fmt, ::std::forward<Ps>(ps)...);
    // The terminating `\0` is not part of the string's length.
    msg.resize(needed);
  }

  ::std::string msg;
};

#ifdef UNITTESTED_FILE
//! [cuda_exception]
UNITTEST("cuda_exception")
{
  auto e = cuda_exception(CUDA_SUCCESS);
  EXPECT(e.what()[0] == 0);
#  if _CCCL_HAS_INCLUDE(<cusolverDn.h>)
  auto e1 = cuda_exception(CUSOLVER_STATUS_ZERO_PIVOT);
  EXPECT(strlen(e1.what()) > 0u);
#  endif
};
//! [cuda_exception]
#endif // UNITTESTED_FILE

namespace reserved
{
template <typename>
struct first_param_impl;

template <typename R, typename P, typename... Ps>
struct first_param_impl<R (*)(P, Ps...)>
{
  using type = P;
};

/*
`reserved::first_param<fun>` is an alias for the type of `fun`'s first parameter.
*/
template <auto f>
using first_param = typename first_param_impl<decltype(f)>::type;
} // namespace reserved

#ifdef UNITTESTED_FILE
UNITTEST("first_param")
{
  extern int test1(int);
  static_assert(::std::is_same_v<reserved::first_param<test1>, int>);
  extern int test2(double, int);
  static_assert(::std::is_same_v<reserved::first_param<test2>, double>);
  extern int test3(int&&);
  static_assert(::std::is_same_v<reserved::first_param<test3>, int&&>);
};
#endif // UNITTESTED_FILE

/**
 * @brief Enforces successful call of CUDA API functions.
 *
 * If `status` is `0`, the function has no effect. Otherwise, the function prints pertinent error information to
 * `stderr` and aborts the program.
 *
 * @tparam T status type, can be `cudaError_t`, `cublasStatus_t`, or `cusolverStatus_t`
 * @param status status value, usually the result of a CUDA API call
 * @param loc location of the call, defaulted
 *
 * @paragraph example Example
 * @snippet this cuda_safe_call
 */
template <typename T>
void cuda_safe_call(const T status, const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
{
  // Common early exit test for all cases
  if (status == 0)
  {
    return;
  }
  fprintf(stderr, "%s\n", cuda_exception(status, loc).what());
  abort();
}

#ifdef UNITTESTED_FILE
UNITTEST("cuda_safe_call")
{
  //! [cuda_safe_call]
  cuda_safe_call(CUDA_SUCCESS); // no effect
  int dev;
  cuda_safe_call(cudaGetDevice(&dev)); // continue execution if the call is successful
  if (false)
  {
    cuda_safe_call(CUDA_ERROR_INVALID_VALUE); // would abort application if called
  }
  //! [cuda_safe_call]
};
#endif // UNITTESTED_FILE

/**
 * @brief Throws a `cuda_exception` if the given `status` is an error code
 *
 * @tparam Status CUDA error code type, such as `cudaError_t`, `cublasStatus_t`, or `cusolverStatus_t`
 * @param status CUDA error code value, usually the result of a CUDA API call
 * @param loc location of the call, defaulted
 *
 * The typical usage is to place a CUDA function call inside `cuda_try`, i.e. `cuda_try(cudaFunc(args))` (the
 * same way `cuda_safe_call` would be called). For example, `cuda_try(cudaCreateStream(&stream))` is equivalent to
 * `cudaCreateStream(&stream)`, with the note that the former call throws an exception in case of error.
 *
 * @paragraph example Example
 * @snippet this cuda_try1
 */
template <typename Status>
void cuda_try(Status status, const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
{
  if (status)
  {
#if _CCCL_HAS_EXCEPTIONS()
    throw cuda_exception(status, loc);
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
    ::cuda::std::terminate();
#endif // !_CCCL_HAS_EXCEPTIONS()
  }
}

#ifdef UNITTESTED_FILE
UNITTEST("cuda_try1")
{
  //! [cuda_try1]
  cuda_try(CUDA_SUCCESS); // no effect, returns CUDA_SUCCESS
  int dev;
  cuda_try(cudaGetDevice(&dev)); // equivalent to the line above
  try
  {
    cuda_try(CUDA_ERROR_INVALID_VALUE); // would abort application if called
  }
  catch (...)
  {
    // This point will be reached
    return;
  }
  EXPECT(false, "Should not get here.");
  //! [cuda_try1]
};
#endif // UNITTESTED_FILE

/**
 * @brief Calls a CUDA function and throws a `cuda_exception` in case of failure.
 *
 * @tparam fun Name of the CUDA function to invoke, cannot be the name of an overloaded function
 * @tparam Ps Parameter types to be forwarded
 * @param ps Arguments to be forwarded
 * @return `auto` (see below)
 *
 * In this overload of `cuda_try`, the function name is passed explicitly as a template argument and also the first
 * argument is omitted, as in `cuda_try<cudaCreateStream>()`. `cuda_try` will create a temporary  of the appropriate
 * type internally, call the specified CUDA function with the address of that temporary, and then return the temporary.
 * For example, in the call `cuda_try<cudaCreateStream>()`, the created stream object will be returned. That way you can
 * write `auto stream = cuda_try<cudaCreateStream>();` instead of `cudaStream_t stream; cudaCreateStream(&stream);`.
 * This invocation mode relies on the convention used by many CUDA functions with output parameters of specifying them
 * in the first parameter position.
 *
 * Limitations: Does not work with overloaded functions.
 *
 * @paragraph example Example
 * @snippet this cuda_try2
 */
template <auto fun, typename... Ps>
auto cuda_try(Ps&&... ps)
{
  if constexpr (::std::is_invocable_v<decltype(fun), Ps...>)
  {
    cuda_try(fun(::std::forward<Ps>(ps)...));
  }
  else
  {
    ::std::remove_pointer_t<reserved::first_param<fun>> result{};
    cuda_try(fun(&result, ::std::forward<Ps>(ps)...));
    return result;
  }
}

#ifdef UNITTESTED_FILE
UNITTEST("cuda_try2")
{
  //! [cuda_try2]
  int dev = cuda_try<cudaGetDevice>(); // continue execution if the call is successful
  cuda_try(cudaGetDevice(&dev)); // equivalent to the line above
  //! [cuda_try2]
};
#endif // UNITTESTED_FILE

// Unused, keep for later
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
#  define OVERLOADS_UNUSED(f)           \
    ba7b8453f262e429575e23dcb2192b33(   \
      a2bce6d11e8033f5c8d9c9442849656c, \
      f(::std::forward<decltype(a2bce6d11e8033f5c8d9c9442849656c)>(a2bce6d11e8033f5c8d9c9442849656c)...))
// Unused, keep for later
#  define ba7b8453f262e429575e23dcb2192b33(a, fun_of_a)                   \
    [&](auto&&... a) noexcept(noexcept(fun_of_a)) -> decltype(fun_of_a) { \
      return fun_of_a;                                                    \
    }

// Unused, keep for later
#  define CUDATRY_UNUSED(fun)         \
    a838e9c10e0ded64dff84e7b679d2342( \
      (fun), a2bce6d11e8033f5c8d9c9442849656c, cca0b395150985cb1c6ab3f8032edafa, fef8664203d67fe27b0434c87ce346fb)
// Unused, keep for later
#  define a838e9c10e0ded64dff84e7b679d2342(f, a, status, result)                   \
    [&](auto&&... a) {                                                             \
      if constexpr (::std::is_invocable_v<decltype(OVERLOADS(f)), decltype(a)...>) \
      {                                                                            \
        ::cuda::experimental::stf::cuda_try(f(::std::forward<decltype(a)>(a)...)); \
      }                                                                            \
      else                                                                         \
      {                                                                            \
        ::std::remove_pointer_t<reserved::first_param<f>> result;                  \
        if (auto status = f(&result, ::std::forward<decltype(a)>(a)...))           \
        {                                                                          \
          throw ::cuda::experimental::stf::cuda_exception(status);                 \
        }                                                                          \
        return result;                                                             \
      }                                                                            \
    } CUDATRY_ACCEPTS_ONLY_FUNCTION_NAMES
// Unused, keep for later
#  define CUDATRY_ACCEPTS_ONLY_FUNCTION_NAMES_UNUSED(...) (__VA_ARGS__)
#endif // !_CCCL_DOXYGEN_INVOKED

} // namespace cuda::experimental::stf
