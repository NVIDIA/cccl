//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___PSTL_IOTA_H
#define _CUDA___PSTL_IOTA_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/__iterator/strided_iterator.h>
#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/not_fn.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/incrementable_traits.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__numeric/iota.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_arithmetic.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__type_traits/is_integral.h>
#  include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/transform.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
_CCCL_CONCEPT __can_operator_plus_integral = _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __val, const uint64_t __index)(
  requires(::cuda::std::is_convertible_v<decltype(__val + __index), _Tp>));

template <class _Tp>
_CCCL_CONCEPT __can_operator_plus_conversion = _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __val, const uint64_t __index)(
  requires(::cuda::std::is_convertible_v<decltype(__val + static_cast<_Tp>(__index)), _Tp>));

template <class _Tp>
struct __iota_init_fn
{
  _Tp __init_;

  _CCCL_API constexpr __iota_init_fn(const _Tp& __init) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Tp>)
      : __init_(__init)
  {}

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE _Tp constexpr operator()(const uint64_t __index) const
  {
#  if _LIBCUDACXX_HAS_NVFP16()
    // We cannot rely on operator+ and constructors from integers to be available for the extended fp types
    if constexpr (::cuda::std::is_same_v<_Tp, __half>)
    {
      return ::__hadd(__init_, ::__ull2half_rn(__index));
    }
    else
#  endif // _LIBCUDACXX_HAS_NVFP16()
#  if _LIBCUDACXX_HAS_NVBF16()
      if constexpr (::cuda::std::is_same_v<_Tp, __nv_bfloat16>)
    {
      return ::__hadd(__init_, ::__ull2bfloat16_rn(__index));
    }
    else
#  endif // _LIBCUDACXX_HAS_NVBF16()
      if constexpr (::cuda::std::is_arithmetic_v<_Tp>)
      { // avoid warnings about integer conversions
        return static_cast<_Tp>(__init_ + static_cast<_Tp>(__index));
      }
      else if constexpr (__can_operator_plus_integral<_Tp>)
      {
        return __init_ + __index;
      }
      else if constexpr (__can_operator_plus_conversion<_Tp>)
      {
        return __init_ + static_cast<_Tp>(__index);
      }
      else
      {
        static_assert(::cuda::std::__always_false_v<_Tp>,
                      "cuda::iota(iter, iter, init) requires that T supports operator+");
      }
  }
};

template <class _Tp>
_CCCL_CONCEPT __can_operator_plus_times_integral = _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __val, const uint64_t __index)(
  requires(::cuda::std::is_convertible_v<decltype(__val + __val * __index), _Tp>));

template <class _Tp>
_CCCL_CONCEPT __can_operator_plus_times_conversion =
  _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __val, const uint64_t __index) //
  (requires(::cuda::std::is_convertible_v<decltype(__val + __val * static_cast<_Tp>(__index)), _Tp>));

template <class _Tp>
struct __iota_init_step_fn
{
  _Tp __init_;
  _Tp __step_;

  _CCCL_API constexpr __iota_init_step_fn(const _Tp& __init,
                                          const _Tp& __step) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Tp>)
      : __init_(__init)
      , __step_(__step)
  {}

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE constexpr _Tp operator()(const uint64_t __index) const
  {
#  if _LIBCUDACXX_HAS_NVFP16()
    // We cannot rely on operator+ and constructors from integers to be available for the extended fp types
    if constexpr (::cuda::std::is_same_v<_Tp, __half>)
    {
      return ::__hadd(__init_, ::__hmul(__step_, ::__ull2half_rn(__index)));
    }
    else
#  endif // _LIBCUDACXX_HAS_NVFP16()
#  if _LIBCUDACXX_HAS_NVBF16()
      if constexpr (::cuda::std::is_same_v<_Tp, __nv_bfloat16>)
    {
      return ::__hadd(__init_, ::__hmul(__step_, ::__ull2bfloat16_rn(__index)));
    }
    else
#  endif // _LIBCUDACXX_HAS_NVBF16()
      if constexpr (::cuda::std::is_arithmetic_v<_Tp>)
      { // avoid warnings about integer conversions
        return static_cast<_Tp>(__init_ + __step_ * static_cast<_Tp>(__index));
      }
      else if constexpr (__can_operator_plus_times_integral<_Tp>)
      {
        return __init_ + __step_ * __index;
      }
      else if constexpr (__can_operator_plus_times_conversion<_Tp>)
      {
        return __init_ + __step_ * static_cast<_Tp>(__index);
      }
      else
      {
        static_assert(::cuda::std::__always_false_v<_Tp>,
                      "cuda::iota(iter, iter, init, step) requires that T supports operator+ and operator*");
      }
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _InputIterator, class _Tp = ::cuda::std::iter_value_t<_InputIterator>)
_CCCL_REQUIRES(
  ::cuda::std::__has_forward_traversal<_InputIterator> _CCCL_AND ::cuda::std::is_execution_policy_v<_Policy>)
_CCCL_HOST_API void
iota([[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last, const _Tp& __init = _Tp{})
{
  static_assert(::cuda::std::indirectly_writable<_InputIterator, _Tp>,
                "cuda::iota requires InputIterator to be indirectly writable with T");

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__transform, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::iota");

    if (__first == __last)
    {
      return;
    }

    // Note: using a different offset type than uint64_t degrades performance considerably for larger integer types
    const auto __count = static_cast<uint64_t>(::cuda::std::distance(__first, __last));
    // For whatever reason __iota_init_step_fn is much faster for int64_t and __int128
    if constexpr (::cuda::std::is_arithmetic_v<_Tp>)
    {
      (void) __dispatch(
        __policy,
        ::cuda::counting_iterator<uint64_t>{0},
        ::cuda::counting_iterator<uint64_t>{__count},
        ::cuda::std::move(__first),
        __iota_init_step_fn{__init, _Tp{1}});
    }
    else
    {
      (void) __dispatch(
        __policy,
        ::cuda::counting_iterator<uint64_t>{0},
        ::cuda::counting_iterator<uint64_t>{static_cast<uint64_t>(__count)},
        ::cuda::std::move(__first),
        __iota_init_fn{__init});
    }
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Policy>, "Parallel cuda::iota requires at least one selected backend");
    return ::cuda::iota(::cuda::std::move(__first), ::cuda::std::move(__last), __init);
  }
}

_CCCL_TEMPLATE(class _Policy, class _InputIterator, class _Tp = ::cuda::std::iter_value_t<_InputIterator>)
_CCCL_REQUIRES(
  ::cuda::std::__has_forward_traversal<_InputIterator> _CCCL_AND ::cuda::std::is_execution_policy_v<_Policy>)
_CCCL_HOST_API void
iota([[maybe_unused]] const _Policy& __policy,
     _InputIterator __first,
     _InputIterator __last,
     const _Tp& __init,
     const _Tp& __step)
{
  static_assert(::cuda::std::indirectly_writable<_InputIterator, _Tp>,
                "cuda::iota requires InputIterator to be indirectly writable with T");

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__transform, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::iota");

    if (__first == __last)
    {
      return;
    }

    // Note: using a different offset type than uint64_t degrades performance considerably for larger integer types
    const auto __count = static_cast<uint64_t>(::cuda::std::distance(__first, __last));
    (void) __dispatch(
      __policy,
      ::cuda::counting_iterator<uint64_t>{0},
      ::cuda::counting_iterator<uint64_t>{__count},
      ::cuda::std::move(__first),
      __iota_init_step_fn{__init, __step});
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Policy>, "Parallel cuda::iota requires at least one selected backend");
    // TODO(miscco): Consider adding that overload to serial iota
    return ::cuda::iota(::cuda::std::move(__first), ::cuda::std::move(__last), __init /*, __step*/);
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___PSTL_IOTA_H
