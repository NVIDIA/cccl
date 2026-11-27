//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_DISPATCH_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_DISPATCH_CUH

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/coop/__group/tags.cuh>
#include <cuda/experimental/coop/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED

namespace cuda::experimental::coop
{
template <class _Group, class _MappingResult = typename _Group::__mapping_result_type>
using __dispatch_view_t _CCCL_NODEBUG =
  group_view<typename _Group::unit_level, __viewed_group_t<_Group>, _MappingResult>;

template <::cuda::std::size_t _Count, ::cuda::std::size_t _GroupRank = ::cuda::std::dynamic_extent, class _Group>
[[nodiscard]] _CCCL_DEVICE_API auto __make_dispatch_view(const _Group& __group)
{
  using _MappingResult = typename _Group::__mapping_result_type;
  using _SpecializedMappingResult =
    __mapping_result<__uniform_static_unit_count<_MappingResult::static_group_count(), _Count>,
                     _MappingResult::is_always_exhaustive(),
                     _MappingResult::is_always_contiguous(),
                     _GroupRank>;

  const auto& __mapping = __group.__mapping_result();
  return __dispatch_view_t<_Group, _SpecializedMappingResult>{
    __private_tag{},
    __group,
    _SpecializedMappingResult{
      __mapping.group_count(),
      __mapping.group_rank(),
      __mapping.unit_count(),
      __mapping.unit_rank(),
      __mapping.lane_mask()}};
}

// Number distinct unit counts in order of first appearance so equal-sized groups dispatch to the same callback.
template <class _MappingResult>
struct __dispatch_unit_counts
{
  ::cuda::std::size_t __size_ = 0;
  ::cuda::std::size_t __counts_[_MappingResult::static_group_count()]{};
  ::cuda::std::uint32_t __indices_[_MappingResult::static_group_count()]{};

  _CCCL_DEVICE_API constexpr __dispatch_unit_counts() noexcept
  {
    for (::cuda::std::size_t __rank = 0; __rank < _MappingResult::static_group_count(); ++__rank)
    {
      const auto __count            = _MappingResult::static_unit_count(__rank);
      ::cuda::std::uint32_t __index = 0;

      while (__index < __size_ && __counts_[__index] != __count)
      {
        ++__index;
      }
      if (__index == __size_)
      {
        __counts_[__size_++] = __count;
      }
      __indices_[__rank] = __index;
    }
  }

  template <unsigned _Bits>
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto __pack_indices() const noexcept
  {
    constexpr auto __per_word = 32 / _Bits;
    constexpr auto __nwords   = ::cuda::ceil_div(_MappingResult::static_group_count(), __per_word);
    ::cuda::std::array<::cuda::std::uint32_t, __nwords> __words{};
    for (::cuda::std::size_t __i = 0; __i < _MappingResult::static_group_count(); ++__i)
    {
      __words[__i / __per_word] |= __indices_[__i] << ((__i % __per_word) * _Bits);
    }
    return __words;
  }
};

template <::cuda::std::size_t _Rank, class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __dispatch_case(by_group_rank_t, const _Group& __group, _Fn&& __fn)
{
  using _MappingResult = typename _Group::__mapping_result_type;
  return ::cuda::std::forward<_Fn>(__fn)(
    ::cuda::experimental::coop::__make_dispatch_view<_MappingResult::static_unit_count(_Rank), _Rank>(__group));
}

template <::cuda::std::size_t _Index, class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __dispatch_case(by_unit_count_t, const _Group& __group, _Fn&& __fn)
{
  constexpr __dispatch_unit_counts<typename _Group::__mapping_result_type> __cases{};
  return ::cuda::std::forward<_Fn>(__fn)(
    ::cuda::experimental::coop::__make_dispatch_view<__cases.__counts_[_Index]>(__group));
}

template <::cuda::std::size_t _Index, ::cuda::std::size_t _Size>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t
__dispatch_case(const ::cuda::std::array<::cuda::std::uint32_t, _Size>& __words) noexcept
{
  return __words[_Index];
}

// Select the case for the runtime index. The conditional operator
// preserves references and void results, and applies the usual conversions between different result types.
template <::cuda::std::size_t _First, ::cuda::std::size_t _Last, class... _Args>
_CCCL_DEVICE_API decltype(auto) __dispatch_index(::cuda::std::uint32_t __index, _Args&&... __args)
{
  if constexpr (_Last - _First == 1)
  {
    return ::cuda::experimental::coop::__dispatch_case<_First>(::cuda::std::forward<_Args>(__args)...);
  }
  else
  {
    constexpr auto __mid = _First + ((_Last - _First) / 2);
    return __index < __mid
           ? ::cuda::experimental::coop::__dispatch_index<_First, __mid>(__index, ::cuda::std::forward<_Args>(__args)...)
           : ::cuda::experimental::coop::__dispatch_index<__mid, _Last>(__index, ::cuda::std::forward<_Args>(__args)...);
  }
}

template <class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __dispatch(const _Group& __group, by_group_rank_t __policy, _Fn&& __fn)
{
  using _MappingResult         = typename _Group::__mapping_result_type;
  constexpr auto __group_count = _MappingResult::static_group_count();

  if constexpr (_MappingResult::static_group_rank() != ::cuda::std::dynamic_extent || __group_count == 0
                || __group_count == ::cuda::std::dynamic_extent)
  {
    return ::cuda::std::forward<_Fn>(__fn)(__dispatch_view_t<_Group>{__group});
  }
  else
  {
    return ::cuda::experimental::coop::__dispatch_index<0, __group_count>(
      __group.__mapping_result().group_rank(), __policy, __group, ::cuda::std::forward<_Fn>(__fn));
  }
}

// Pack case indices into immediate words so looking up a rank needs no per-mapping device table.
template <class _MappingResult>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t __unit_count_index(::cuda::std::uint32_t __rank) noexcept
{
  constexpr __dispatch_unit_counts<_MappingResult> __cases{};

  if constexpr (__cases.__size_ == _MappingResult::static_group_count())
  {
    return __rank;
  }
  else
  {
    constexpr auto __bits     = ::cuda::std::bit_ceil<unsigned>(::cuda::std::bit_width(__cases.__size_ - 1));
    constexpr auto __per_word = 32 / __bits;
    constexpr auto __words    = __cases.template __pack_indices<__bits>();
    const auto __word = ::cuda::experimental::coop::__dispatch_index<0, __words.size()>(__rank / __per_word, __words);
    return (__word >> ((__rank % __per_word) * __bits)) & (~::cuda::std::uint32_t{0} >> (32 - __bits));
  }
}

template <class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __dispatch(const _Group& __group, by_unit_count_t __policy, _Fn&& __fn)
{
  using _MappingResult = typename _Group::__mapping_result_type;

  if constexpr (_MappingResult::has_uniform_static_unit_count())
  {
    return ::cuda::std::forward<_Fn>(__fn)(__dispatch_view_t<_Group>{__group});
  }
  else
  {
    constexpr __dispatch_unit_counts<_MappingResult> __cases{};

    const auto __rank  = __group.__mapping_result().group_rank();
    const auto __index = ::cuda::experimental::coop::__unit_count_index<_MappingResult>(__rank);

    return ::cuda::experimental::coop::__dispatch_index<0, __cases.__size_>(
      __index, __policy, __group, ::cuda::std::forward<_Fn>(__fn));
  }
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_DISPATCH_CUH
