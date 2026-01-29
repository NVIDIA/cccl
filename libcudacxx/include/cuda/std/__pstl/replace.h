//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_REPLACE_H
#define _CUDA_STD___PSTL_REPLACE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__algorithm/replace.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_comparable.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/transform.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
struct __replace_compare_eq
{
  _Tp __old_value_;

  _CCCL_HOST_API constexpr __replace_compare_eq(const _Tp& __old_value) noexcept(is_nothrow_copy_constructible_v<_Tp>)
      : __old_value_(__old_value)
  {}

  template <class _Up>
  [[nodiscard]] _CCCL_DEVICE_API constexpr bool operator()(const _Up& __value) const
    noexcept(__is_cpp17_nothrow_equality_comparable_v<_Tp, _Up>)
  {
    return static_cast<bool>(__old_value_ == __value);
  }
};

template <class _Tp>
struct __replace_return_value
{
  _Tp __new_value_;

  _CCCL_HOST_API constexpr __replace_return_value(const _Tp& __new_value) noexcept(is_nothrow_copy_constructible_v<_Tp>)
      : __new_value_(__new_value)
  {}

  template <class _Up>
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp operator()(const _Up&) const
    noexcept(is_nothrow_copy_constructible_v<_Tp>)
  {
    return __new_value_;
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _InputIterator, class _Tp = iter_value_t<_InputIterator>)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator> _CCCL_AND is_execution_policy_v<_Policy>)
_CCCL_HOST_API void replace(
  [[maybe_unused]] const _Policy& __policy,
  _InputIterator __first,
  _InputIterator __last,
  const _Tp& __old_value,
  const _Tp& __new_value)
{
  static_assert(__is_cpp17_equality_comparable_v<_Tp, iter_reference_t<_InputIterator>>,
                "cuda::std::replace requires T to be comparable with iter_reference_t<InputIterator>");

  if (__first == __last)
  {
    return;
  }

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__transform, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    (void) __dispatch(
      __policy,
      __first,
      ::cuda::std::move(__last),
      ::cuda::std::move(__first),
      __replace_return_value{__new_value},
      __replace_compare_eq{__old_value});
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::replace requires at least one selected backend");
    ::cuda::std::replace(::cuda::std::move(__first), ::cuda::std::move(__last), __old_value, __new_value);
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_REPLACE_H
