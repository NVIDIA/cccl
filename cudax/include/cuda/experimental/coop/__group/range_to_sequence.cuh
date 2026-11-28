//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_RANGE_TO_SEQUENCE_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_RANGE_TO_SEQUENCE_CUH

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/narrow.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__utility/constant_wrapper.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED

namespace cuda::experimental::coop
{
// Materialize a compile-time range into a sequence without requiring random-access iterators.
template <class _Wrapper, ::cuda::std::size_t... _Is>
[[nodiscard]]
_CCCL_DEVICE_API constexpr auto __range_to_index_sequence(::cuda::std::index_sequence<_Is...>)
  -> ::cuda::std::index_sequence<
    ::cuda::narrow<::cuda::std::size_t>(*::cuda::std::next(::cuda::std::ranges::begin(_Wrapper::value), _Is))...>;

_CCCL_TEMPLATE(auto _Range, class _Tp)
_CCCL_REQUIRES(
  ::cuda::std::ranges::sized_range<decltype((::cuda::std::constant_wrapper<_Range, _Tp>::value))>
    _CCCL_AND ::cuda::std::ranges::forward_range<decltype((::cuda::std::constant_wrapper<_Range, _Tp>::value))>
      _CCCL_AND ::cuda::std::is_convertible_v<
        ::cuda::std::ranges::range_value_t<decltype((::cuda::std::constant_wrapper<_Range, _Tp>::value))>,
        ::cuda::std::size_t>)
_CCCL_DEVICE_API auto __to_index_sequence(const ::cuda::std::constant_wrapper<_Range, _Tp>&)
  -> decltype(::cuda::experimental::coop::__range_to_index_sequence<::cuda::std::constant_wrapper<_Range, _Tp>>(
    ::cuda::std::make_index_sequence<::cuda::std::ranges::size(::cuda::std::constant_wrapper<_Range, _Tp>::value)>{}));

template <class _Int, _Int... _Values>
_CCCL_DEVICE_API auto __to_index_sequence(const ::cuda::std::integer_sequence<_Int, _Values...>&)
  -> ::cuda::std::index_sequence<::cuda::narrow<::cuda::std::size_t>(_Values)...>;

template <class _StaticRange>
using __to_index_sequence_t _CCCL_NODEBUG =
  decltype(::cuda::experimental::coop::__to_index_sequence(::cuda::std::declval<const _StaticRange&>()));
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_RANGE_TO_SEQUENCE_CUH
