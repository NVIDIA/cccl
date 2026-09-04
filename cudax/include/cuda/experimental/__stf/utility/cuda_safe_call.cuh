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
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/source_location>

#include <cuda/experimental/__stf/utility/exception_policy.cuh>
#include <cuda/experimental/__stf/utility/source_location.cuh>
#include <cuda/experimental/__stf/utility/unittest.cuh>

#include <cstdlib>
#include <exception>

#include <cuda.h>
#include <cuda_occupancy.h>
#include <cuda_runtime.h>

#if __has_include(<cusolverDn.h>)
#  include <cusolverDn.h>
#endif

namespace cuda::experimental::stf
{
#if __has_include(<cusolverDn.h>)
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
#if __has_include(<cublas_v2.h>)
                    && CUBLAS_STATUS_SUCCESS == 0
#endif
#if __has_include(<cusolverDn.h>)
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

#if __has_include(<cusolverDn.h>)
    if constexpr (::cuda::std::is_same_v<T, cusolverStatus_t>)
    {
      format("%s(%u) [device %d] CUSOLVER error in call %s: %s.",
             loc.file_name(),
             loc.line(),
             dev,
             loc.function_name(),
             cusolverGetErrorString(status));
    }
    else
#endif // __has_include(<cusolverDn.h>)
#if __has_include(<cublas_v2.h>)
      if constexpr (::cuda::std::is_same_v<T, cublasStatus_t>)
    {
      format("%s(%u) [device %d] CUBLAS error in %s: %s.",
             loc.file_name(),
             loc.line(),
             dev,
             loc.function_name(),
             cublasGetStatusString(status));
    }
    else
#endif // __has_include(<cublas_v2.h>)
      if constexpr (::cuda::std::is_same_v<T, cudaOccError>)
      {
        format("%s(%u) [device %d] CUDA OCC error in %s: %s.",
               loc.file_name(),
               loc.line(),
               dev,
               loc.function_name(),
               cudaGetErrorString(cudaErrorInvalidConfiguration));
      }
      else if constexpr (::cuda::std::is_same_v<T, CUresult>)
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
        static_assert(::cuda::std::is_same_v<T, cudaError_t>, "Error: not a CUDA status.");
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
    ::std::snprintf(&msg[0], msg.capacity(), fmt, ::cuda::std::forward<Ps>(ps)...);
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
#  if __has_include(<cusolverDn.h>)
  auto e1 = cuda_exception(CUSOLVER_STATUS_ZERO_PIVOT);
  EXPECT(strlen(e1.what()) > 0u);
#  endif
};
//! [cuda_exception]
#endif // UNITTESTED_FILE

namespace reserved
{
// Placeholder for "this function has no such parameter". Nothing converts to it, so a
// candidate overload keyed on it is silently non-viable rather than a hard error.
struct no_param
{};

template <typename, typename = void>
struct first_param_impl
{
  using type = no_param;
};

template <typename R, typename P, typename... Ps>
struct first_param_impl<R (*)(P, Ps...)>
{
  using type = P;
};

template <typename, typename = void>
struct second_param_impl
{
  using type = no_param;
};

template <typename R, typename P0, typename P1, typename... Ps>
struct second_param_impl<R (*)(P0, P1, Ps...)>
{
  using type = P1;
};

template <typename...>
struct last_param_impl;

template <typename P>
struct last_param_impl<P>
{
  using type = P;
};

template <typename P, typename... Ps>
struct last_param_impl<P, Ps...> : last_param_impl<Ps...>
{};

template <typename, typename = void>
struct function_last_param_impl
{
  using type = no_param;
};

template <typename R, typename... Ps>
struct function_last_param_impl<R (*)(Ps...)>
{
  using type = typename last_param_impl<Ps...>::type;
};

// `R (*)()` has no last parameter: the primary template's no_param applies.
template <typename R>
struct function_last_param_impl<R (*)()>
{
  using type = no_param;
};

// The pointee of a synthesized output parameter: `T*` -> `T`, anything else -> no_param.
template <typename>
struct out_param_impl
{
  using type = no_param;
};

template <typename T>
struct out_param_impl<T*>
{
  using type = T;
};

/*
`reserved::first_param<fun>` is an alias for the type of `fun`'s first parameter.
*/
template <auto f>
using first_param = typename first_param_impl<decltype(f)>::type;

/*
`reserved::last_param<fun>` is an alias for the type of `fun`'s last parameter.
*/
template <auto f>
using last_param = typename function_last_param_impl<decltype(f)>::type;

/*
`reserved::second_param<fun>` is an alias for the type of `fun`'s second parameter, or
`no_param` when it has fewer than two.
*/
template <auto f>
using second_param = typename second_param_impl<decltype(f)>::type;

/*
`reserved::out_param<T>` is `T`'s pointee when `T` is a pointer, else `no_param`. Used to
name the type of a synthesized output argument.
*/
template <typename T>
using out_param = typename out_param_impl<T>::type;

template <typename...>
inline constexpr bool dependent_false = false;
} // namespace reserved

#ifdef UNITTESTED_FILE
UNITTEST("first_last_param")
{
  extern int test1(int);
  static_assert(::cuda::std::is_same_v<reserved::first_param<test1>, int>);
  static_assert(::cuda::std::is_same_v<reserved::last_param<test1>, int>);
  extern int test2(double, int);
  static_assert(::cuda::std::is_same_v<reserved::first_param<test2>, double>);
  static_assert(::cuda::std::is_same_v<reserved::last_param<test2>, int>);
  extern int test3(int&&);
  static_assert(::cuda::std::is_same_v<reserved::first_param<test3>, int&&>);
  static_assert(::cuda::std::is_same_v<reserved::last_param<test3>, int&&>);
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
 * @snippet this cuda_try1
 */
template <typename Status>
void cuda_try(Status status, const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
{
  if (status)
  {
    // _CCCL_THROW itself terminates (with a report) when exceptions are disabled.
    _CCCL_THROW(cuda_exception, status, loc);
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
 * @brief Calls a CUDA function with optional output-parameter inference and throws a `cuda_exception` on failure.
 *
 * @tparam fun The CUDA function to invoke. Must not be an overloaded name (templated overloads in
 *             `cuda_runtime.h` such as `cudaMalloc`/`cudaMallocHost`/`cudaMallocAsync` therefore do not work).
 * @tparam Ps  Argument types deduced from @p ps.
 * @param ps   Arguments forwarded to @p fun.
 * @return     `void` if @p fun does not have a synthesized output parameter (see below); otherwise the value of the
 *             synthesized output parameter.
 *
 * Calls @p fun and translates a non-zero CUDA status into a thrown `cuda_exception`. Three call shapes are
 * supported, selected at compile time in the following order:
 *
 *  1. **Direct form.** If `fun(ps...)` is invocable, the call is made directly and the status is checked. The
 *     return type is `void`.
 *
 *  2. **First-parameter output form.** Otherwise, if @p fun's first parameter is a non-`const` pointer (an output
 *     pointer by CUDA convention) and `fun(&result, ps...)` is invocable, a temporary `result` of the pointee type
 *     is value-initialized, the call is made, and `result` is returned. This matches CUDA APIs like
 *     `cudaStreamCreate(cudaStream_t*)`, `cudaGraphAddEmptyNode(cudaGraphNode_t*, ...)`, and
 *     `cudaDeviceCanAccessPeer(int*, ...)`.
 *
 *  3. **Last-parameter output form.** Otherwise, if @p fun's last parameter is a non-`const` pointer and
 *     `fun(ps..., &result)` is invocable, the temporary is appended instead and returned. This matches CUDA APIs
 *     like `cuStreamGetId(CUstream, unsigned long long*)` and `cuCtxGetId(CUcontext, unsigned long long*)`.
 *
 * If none of the three forms apply, compilation fails with a `static_assert` that says no valid invocation form
 * exists for the given function and arguments.
 *
 * **Ambiguity rejection.** When @p fun has non-`const` pointer parameters in both the first and last positions,
 * the same user arguments can satisfy both the first- and last-parameter output forms with different effects
 * (for example, `cudaMemGetInfo(size_t* free, size_t* total)` called with one user-supplied `size_t*`). In that
 * case a `static_assert` rejects the call with the message:
 *
 *     "Ambiguous cuda_try: both first- and last-output forms apply; call the function explicitly to disambiguate."
 *
 * The single zero-argument case (`cuda_try<fun>()`) is exempt because the synthesized call `fun(&result)` is
 * identical for both interpretations.
 *
 * @par Examples
 * @code
 * auto dev = cuda_try<cudaGetDevice>();                          // first-parameter output form
 * auto id  = cuda_try<cuStreamGetId>(some_cu_stream);            // last-parameter output form
 * cuda_try<cudaSetDevice>(0);                                    // direct form, returns void
 * cuda_try(cudaSetDevice(0));                                    // equivalent runtime-status overload
 * @endcode
 *
 * @par Limitations
 *  - Overloaded functions are not supported. CUDA's templated wrappers in `cuda_runtime.h` (e.g. `cudaMalloc`,
 *    `cudaMallocHost`, `cudaMallocManaged`, `cudaMallocAsync`, `cudaHostAlloc`) are overloads and must be invoked
 *    using the runtime-status overload, e.g. `cuda_try(cudaMalloc(&p, n))`.
 *  - The synthesized output parameter must be a non-`const` pointer; in/out parameters expressed as
 *    pointer-to-existing-storage are not synthesized and must be passed explicitly.
 *  - In ambiguous cases (see above) the call must be written explicitly via the runtime-status overload.
 *
 * @snippet this cuda_try2
 */
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace reserved
{
// Shared engine for the introspected call forms of `cuda_try<fun>` and `cuda_safe_call<fun>`:
// selects among the direct / first-output / last-output shapes at compile time and hands the
// raw status to `check` (a stateless callable: throw for cuda_try, report-and-abort for
// cuda_safe_call). `loc` is the user's call site, captured by the public fronts (see below).
template <auto fun, typename Check, typename... Ps>
auto checked_api_call(Check check, const ::cuda::std::source_location loc, Ps&&... ps)
{
  constexpr bool direct_form = ::cuda::std::is_invocable_v<decltype(fun), Ps...>;

  constexpr bool first_output_form =
    ::cuda::std::is_pointer_v<reserved::first_param<fun>>
    && !::cuda::std::is_const_v<reserved::out_param<reserved::first_param<fun>>>
    && ::cuda::std::is_invocable_v<decltype(fun), reserved::out_param<reserved::first_param<fun>>*, Ps...>;

  constexpr bool last_output_form =
    ::cuda::std::is_pointer_v<reserved::last_param<fun>>
    && !::cuda::std::is_const_v<reserved::out_param<reserved::last_param<fun>>>
    && ::cuda::std::is_invocable_v<decltype(fun), Ps..., reserved::out_param<reserved::last_param<fun>>*>;

  // When no user args are supplied, the first- and last-output forms produce the same call
  // `fun(&result)`, so they are not ambiguous. Otherwise, both matching is a real ambiguity.
  static_assert(!(first_output_form && last_output_form) || sizeof...(Ps) == 0,
                "Ambiguous cuda_try/cuda_safe_call: both first- and last-output forms apply; "
                "call the function explicitly to disambiguate.");

  if constexpr (direct_form)
  {
    check(fun(::cuda::std::forward<Ps>(ps)...), loc);
  }
  else if constexpr (first_output_form)
  {
    reserved::out_param<reserved::first_param<fun>> result{};
    check(fun(&result, ::cuda::std::forward<Ps>(ps)...), loc);
    return result;
  }
  else if constexpr (last_output_form)
  {
    reserved::out_param<reserved::last_param<fun>> result{};
    check(fun(::cuda::std::forward<Ps>(ps)..., &result), loc);
    return result;
  }
  else
  {
    static_assert(reserved::dependent_false<Ps...>,
                  "No valid cuda_try/cuda_safe_call invocation form for this function.");
  }
}
} // namespace reserved
#endif // !_CCCL_DOXYGEN_INVOKED

// The public fronts capture the USER's call site. A single variadic front cannot: a parameter
// pack followed by a defaulted source_location is a non-deduced context. Instead the first
// user argument (when there is one) is taken as `with_location<T>` where T is COMPUTED from
// `fun`'s signature rather than deduced, so the raw argument converts and the conversion
// captures the location. Two argument-taking overloads are needed because the first user
// argument aligns with `fun`'s first parameter for the direct and last-output forms, but with
// its second parameter for the first-output form (where parameter one is the synthesized
// output pointer). `forward<declared type>` preserves the parameter's value category, so
// reference parameters pass through unchanged. Call syntax is unchanged.
template <auto fun>
auto cuda_try(const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
{
  return reserved::checked_api_call<fun>(
    [](auto status, auto l) {
      cuda_try(status, l);
    },
    loc);
}

_CCCL_TEMPLATE(auto fun, typename... Ps)
_CCCL_REQUIRES((
  ::cuda::std::is_invocable_v<decltype(fun), reserved::first_param<fun>, Ps...>
  || ::cuda::std::
    is_invocable_v<decltype(fun), reserved::first_param<fun>, Ps..., reserved::out_param<reserved::last_param<fun>>*>) )
auto cuda_try(with_location<reserved::first_param<fun>> p0, Ps&&... rest)
{
  return reserved::checked_api_call<fun>(
    [](auto status, auto l) {
      cuda_try(status, l);
    },
    p0.loc,
    ::cuda::std::forward<reserved::first_param<fun>>(p0.payload),
    ::cuda::std::forward<Ps>(rest)...);
}

_CCCL_TEMPLATE(auto fun, typename... Ps)
_CCCL_REQUIRES(
  ::cuda::std::
    is_invocable_v<decltype(fun), reserved::out_param<reserved::first_param<fun>>*, reserved::second_param<fun>, Ps...>)
auto cuda_try(with_location<reserved::second_param<fun>> p0, Ps&&... rest)
{
  reserved::out_param<reserved::first_param<fun>> result{};
  cuda_try(
    fun(&result, ::cuda::std::forward<reserved::second_param<fun>>(p0.payload), ::cuda::std::forward<Ps>(rest)...),
    p0.loc);
  return result;
}

/**
 * @brief As @ref cuda_try with the same three call shapes and the same limitations, but a
 * failing status reports and aborts instead of throwing (the `cuda_safe_call` reaction).
 *
 * The two spellings are deliberately parallel so migrating a call site between the aborting
 * and throwing regimes is a one-token change: `cuda_safe_call<f>(a)` <-> `cuda_try<f>(a)`.
 *
 * @snippet this cuda_safe_call2
 */
template <auto fun>
auto cuda_safe_call(const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
{
  return reserved::checked_api_call<fun>(
    [](auto status, auto l) {
      cuda_safe_call(status, l);
    },
    loc);
}

_CCCL_TEMPLATE(auto fun, typename... Ps)
_CCCL_REQUIRES((
  ::cuda::std::is_invocable_v<decltype(fun), reserved::first_param<fun>, Ps...>
  || ::cuda::std::
    is_invocable_v<decltype(fun), reserved::first_param<fun>, Ps..., reserved::out_param<reserved::last_param<fun>>*>) )
auto cuda_safe_call(with_location<reserved::first_param<fun>> p0, Ps&&... rest)
{
  return reserved::checked_api_call<fun>(
    [](auto status, auto l) {
      cuda_safe_call(status, l);
    },
    p0.loc,
    ::cuda::std::forward<reserved::first_param<fun>>(p0.payload),
    ::cuda::std::forward<Ps>(rest)...);
}

_CCCL_TEMPLATE(auto fun, typename... Ps)
_CCCL_REQUIRES(
  ::cuda::std::
    is_invocable_v<decltype(fun), reserved::out_param<reserved::first_param<fun>>*, reserved::second_param<fun>, Ps...>)
auto cuda_safe_call(with_location<reserved::second_param<fun>> p0, Ps&&... rest)
{
  reserved::out_param<reserved::first_param<fun>> result{};
  cuda_safe_call(
    fun(&result, ::cuda::std::forward<reserved::second_param<fun>>(p0.payload), ::cuda::std::forward<Ps>(rest)...),
    p0.loc);
  return result;
}

#ifdef UNITTESTED_FILE
inline cudaError_t test_first_output_param(int* out)
{
  *out = 1;
  return cudaSuccess;
}

inline cudaError_t test_last_output_param(double in, int* out)
{
  *out = static_cast<int>(in);
  return cudaSuccess;
}

UNITTEST("cuda_try2")
{
  //! [cuda_try2]
  int dev = cuda_try<cudaGetDevice>(); // continue execution if the call is successful
  cuda_try(cudaGetDevice(&dev)); // equivalent to the line above
  EXPECT(cuda_try<test_first_output_param>() == 1);
  EXPECT(cuda_try<test_last_output_param>(2.0) == 2);
  //! [cuda_try2]
};

inline cudaError_t test_failing_direct(int)
{
  return cudaErrorInvalidValue;
}

inline cudaError_t test_lvalue_ref_param(int& x)
{
  x = 5;
  return cudaSuccess;
}

UNITTEST("cuda_try location capture")
{
  // The introspected forms report the CALLER's location, not this header's.
  const auto expected_line = ::cuda::std::source_location::current().line() + 3;
  try
  {
    cuda_try<test_failing_direct>(0);
    EXPECT(false, "should have thrown");
  }
  catch (const cuda_exception& e)
  {
    // The call site is in this same header (the test lives here), so the file name cannot
    // discriminate; the LINE can, and that is the whole claim: the report points at the
    // cuda_try call below, not at the implementation lines above.
    const ::std::string msg = e.what();
    EXPECT(msg.find("(" + ::std::to_string(expected_line) + ")") != ::std::string::npos,
           "the report does not point at the caller's line");
  }

  // Reference parameters keep their value category through the wrapper.
  int v = 0;
  cuda_try<test_lvalue_ref_param>(v);
  EXPECT(v == 5);
};

UNITTEST("cuda_safe_call2")
{
  //! [cuda_safe_call2]
  int dev = cuda_safe_call<cudaGetDevice>(); // aborting sibling of cuda_try<cudaGetDevice>()
  cuda_safe_call(cudaGetDevice(&dev)); // equivalent to the line above
  EXPECT(cuda_safe_call<test_first_output_param>() == 1);
  EXPECT(cuda_safe_call<test_last_output_param>(2.0) == 2);
  //! [cuda_safe_call2]
};
#endif // UNITTESTED_FILE

// Unused, keep for later
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
#  define OVERLOADS_UNUSED(f)           \
    ba7b8453f262e429575e23dcb2192b33(   \
      a2bce6d11e8033f5c8d9c9442849656c, \
      f(::cuda::std::forward<decltype(a2bce6d11e8033f5c8d9c9442849656c)>(a2bce6d11e8033f5c8d9c9442849656c)...))
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
#  define a838e9c10e0ded64dff84e7b679d2342(f, a, status, result)                         \
    [&](auto&&... a) {                                                                   \
      if constexpr (::cuda::std::is_invocable_v<decltype(OVERLOADS(f)), decltype(a)...>) \
      {                                                                                  \
        ::cuda::experimental::stf::cuda_try(f(::cuda::std::forward<decltype(a)>(a)...)); \
      }                                                                                  \
      else                                                                               \
      {                                                                                  \
        ::cuda::std::remove_pointer_t<reserved::first_param<f>> result;                  \
        if (auto status = f(&result, ::cuda::std::forward<decltype(a)>(a)...))           \
        {                                                                                \
          _CCCL_THROW(::cuda::experimental::stf::cuda_exception, status);                \
        }                                                                                \
        return result;                                                                   \
      }                                                                                  \
    } CUDATRY_ACCEPTS_ONLY_FUNCTION_NAMES
// Unused, keep for later
#  define CUDATRY_ACCEPTS_ONLY_FUNCTION_NAMES_UNUSED(...) (__VA_ARGS__)
#endif // !_CCCL_DOXYGEN_INVOKED
} // namespace cuda::experimental::stf
