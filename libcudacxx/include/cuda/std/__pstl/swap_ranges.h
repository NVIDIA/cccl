//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_SWAP_RANGES_H
#define _CUDA_STD___PSTL_SWAP_RANGES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__functional/address_stability.h>
#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/__iterator/zip_iterator.h>
#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__algorithm/iter_swap.h>
#  include <cuda/std/__algorithm/swap_ranges.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/incrementable_traits.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__iterator/next.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__type_traits/is_swappable.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/__utility/swap.h>
#  include <cuda/std/tuple>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/for_each_n.h>
#    include <cuda/std/__pstl/cuda/transform.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _InputIterator1, class _InputIterator2>
struct __swap_ranges_iter_swap_fn
{
  _InputIterator1 __first1;
  _InputIterator2 __first2;

  template <class _DifferenceType>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE constexpr void operator()(const _DifferenceType __index) const
  {
    ::cuda::std::iter_swap(__first1 + __index, __first2 + static_cast<iter_difference_t<_InputIterator2>>(__index));
  }
};

struct __swap_ranges_transform_fn
{
  template <class _Tp, class _Up>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE constexpr auto operator()(_Tp __lhs, _Up __rhs) const
  {
    using ::cuda::std::swap;
    swap(__lhs, __rhs);
    return ::cuda::std::tuple{__lhs, __rhs};
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _InputIterator1, class _InputIterator2)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator1> _CCCL_AND __has_forward_traversal<_InputIterator2> _CCCL_AND
                 is_execution_policy_v<_Policy>)
_CCCL_HOST_API _InputIterator2 swap_ranges(
  [[maybe_unused]] const _Policy& __policy, _InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2)
{
  // We can optimize to using DeviceTransform if neither of the iterators specializes `iter_swap` and there is a
  // transform dispatch
  [[maybe_unused]] auto __transform_dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__transform, _Policy>();
  if constexpr (!::cuda::std::__iter_swap::__unqualified_iter_swap<_InputIterator1, _InputIterator2>
                && ::cuda::std::execution::__pstl_can_dispatch<decltype(__transform_dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::swap_ranges");

    if (__first1 == __last1)
    {
      return __first2;
    }

    const auto __count = ::cuda::std::distance(__first1, __last1);
    auto __ret         = ::cuda::std::next(__first2, static_cast<iter_difference_t<_InputIterator2>>(__count));

    auto __zip_first = ::cuda::zip_iterator{__first1, __first2};

    (void) __transform_dispatch(
      __policy,
      ::cuda::std::move(__first1),
      ::cuda::std::move(__last1),
      ::cuda::std::move(__first2),
      ::cuda::std::move(__zip_first),
      __swap_ranges_transform_fn{});
    return __ret;
  }
  else
  {
    [[maybe_unused]] auto __for_each_dispatch =
      ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__for_each_n, _Policy>();
    if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__for_each_dispatch)>)
    {
      _CCCL_NVTX_RANGE_SCOPE("cuda::std::swap_ranges");

      if (__first1 == __last1)
      {
        return __first2;
      }

      const auto __count = ::cuda::std::distance(__first1, __last1);
      auto __ret         = __first2 + static_cast<iter_difference_t<_InputIterator2>>(__count);
      (void) __for_each_dispatch(
        __policy,
        ::cuda::counting_iterator<iter_difference_t<_InputIterator1>>{0},
        __count,
        __swap_ranges_iter_swap_fn<_InputIterator1, _InputIterator2>{
          ::cuda::std::move(__first1), ::cuda::std::move(__first2)});
      return __ret;
    }
    else
    {
      static_assert(__always_false_v<_Policy>,
                    "Parallel cuda::std::swap_ranges requires at least one selected backend");
      return ::cuda::std::swap_ranges(
        ::cuda::std::move(__first1), ::cuda::std::move(__last1), ::cuda::std::move(__first2));
    }
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_SWAP_RANGES_H
