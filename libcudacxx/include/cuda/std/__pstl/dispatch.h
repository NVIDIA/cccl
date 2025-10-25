//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_DISPATCH_H
#define _CUDA_STD___PSTL_DISPATCH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__execution/policy.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_EXECUTION

enum class __pstl_algorithm
{
  // The find_if family
  __find,
  __find_if,
  __any_of,
  __all_of,
  __none_of,
  __is_partitioned,

  // merge family
  // non implemented

  // sort family
  __sort,

  // for_each family
  __for_each_n,
  __fill,
  __fill_n,
  __replace,
  __replace_if,
  __generate,
  __generate_n,

  // transform_reduce and transform_reduce_binary family
  __count_if,
  __count,
  __equal,
  __reduce,

  // transform and transform_binary family
  __replace_copy_if,
  __replace_copy,
  __move,
  __copy,
  __copy_n,
  __rotate_copy,
};

//! @brief tag type to indicate that we cannot dispatch to a parallel algorithm and should run the algorithm serially
struct __pstl_no_dispatch
{};

//! @brief Dispatcher for a given @tparam _Algorith and @tparam _Policy
//! If @class __pstl_dispatch is not specialized by the chosen backend we will fall back to serial execution
template <__pstl_algorithm _Algorithm, __execution_backend _Backend>
struct __pstl_dispatch : public __pstl_no_dispatch
{};

//! @brief Helper variable that detects whether @class __pstl_dispatch has been specialized so that we can
//! dispatch
template <class>
inline constexpr bool __pstl_can_dispatch = false;

template <__pstl_algorithm _Algorithm, __execution_backend _Backend>
inline constexpr bool __pstl_can_dispatch<__pstl_dispatch<_Algorithm, _Backend>> =
  !::cuda::std::is_base_of_v<__pstl_no_dispatch, __pstl_dispatch<_Algorithm, _Backend>>;

//! @brief Top layer dispatcher that returns a concrete dispatch if possible
template <__pstl_algorithm _Algorithm, class _Policy>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __pstl_select_dispatch() noexcept
{
  // First extract the desired backend from the policy
  constexpr __execution_backend __backend = _Policy::__get_backend();

  // If the user requests a unique backends, we must take that
  if constexpr (::cuda::std::execution::__has_unique_backend(__backend))
  {
    return __pstl_dispatch<_Algorithm, __backend>{};
  }

  // If the user allows multiple backends, we can take the first available one
  if constexpr (__backend != __execution_backend::__backend_none)
  {
#if _CCCL_HAS_BACKEND_CUDA()
    if constexpr (::cuda::std::execution::__has_matching_backend(__backend, __execution_backend::__backend_cuda))
    {
      using __dispatch = __pstl_dispatch<_Algorithm, __execution_backend::__backend_cuda>;
      if constexpr (__pstl_can_dispatch<__dispatch>) // only take the backend if it can dispatch the algorithm
      {
        return __dispatch{};
      }
    }
#endif // _CCCL_HAS_BACKEND_CUDA()
#if _CCCL_HAS_BACKEND_OMP()
    if constexpr (::cuda::std::execution::__has_matching_backend(__backend, __execution_backend::__backend_omp))
    {
      using __dispatch = __pstl_dispatch<_Algorithm, __execution_backend::__backend_omp>;
      if constexpr (__pstl_can_dispatch<__dispatch>)
      {
        return __dispatch{};
      }
    }
#endif // _CCCL_HAS_BACKEND_OMP()
#if _CCCL_HAS_BACKEND_TBB()
    if constexpr (::cuda::std::execution::__has_matching_backend(__backend, __execution_backend::__backend_tbb))
    {
      using __dispatch = __pstl_dispatch<_Algorithm, __execution_backend::__backend_tbb>;
      if constexpr (__pstl_can_dispatch<__dispatch>)
      {
        return __dispatch{};
      }
    }
#endif // _CCCL_HAS_BACKEND_TBB()
  }
  // If the user did not specify any backend we can chose a favorite
  else /* if constexpr (__backend == __execution_backend::__backend_none) */
  {
    // Depending on the execution policy we can chose a backend
    [[maybe_unused]] constexpr __execution_policy __pol = _Policy::__get_policy();
#if _CCCL_HAS_BACKEND_CUDA()
    if constexpr (__pol == __execution_policy::__parallel_unsequenced)
    {
      return __pstl_dispatch<_Algorithm, __execution_backend::__backend_cuda>{};
    }
    else
#endif // _CCCL_HAS_BACKEND_CUDA()
    {
      // No dispatch found, return invalid to signal serial execution
      return __pstl_dispatch<_Algorithm, __execution_backend::__backend_none>{};
    }
  }
}

_CCCL_END_NAMESPACE_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___PSTL_DISPATCH_H
