//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_LABELED_PARTITION_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_LABELED_PARTITION_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__bit/bit_trunc.h>
#include <cuda/__fwd/hierarchy.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__warp/lane_mask.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API auto __extract_label(_Tp __v)
{
  if constexpr (::cuda::std::is_enum_v<_Tp>)
  {
    return ::cuda::experimental::coop::__extract_label(::cuda::std::to_underlying(__v));
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, bool>)
  {
    return __v;
  }
  else
  {
    static_assert(::cuda::std::is_integral_v<_Tp>,
                  "cuda::experimental::coop::labeled_partition requires the label to be an integral");
    return static_cast<::cuda::std::__make_nbit_uint_t<::cuda::std::__num_bits_v<_Tp>>>(__v);
  }
}

template <class _Fn>
class labeled_partition
{
  static_assert(::cuda::std::is_move_constructible_v<_Fn>, "_Fn must be move constructible");

  _Fn __fn_;

public:
  _CCCL_DEVICE_API explicit labeled_partition(_Fn __fn)
      : __fn_(::cuda::std::move(__fn))
  {}

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result)
  {
    static_assert(::cuda::std::is_same_v<_Unit, thread_level>, "labeled_partition can only group threads");
    static_assert(::cuda::std::is_same_v<typename _ParentGroup::level_type, warp_level>,
                  "labeled_partition can be only used within warp_level");

    using ::cuda::std::uint32_t;
    using ::cuda::std::uint64_t;

    using _Label = decltype(::cuda::experimental::coop::__extract_label(__fn_(__prev_mapping_result)));

    if constexpr (::cuda::std::is_same_v<bool, _Label>)
    {
      constexpr auto __static_prev_ngroups = _PrevMappingResult::static_group_count();
      constexpr auto __static_ngroups =
        (__static_prev_ngroups != ::cuda::std::dynamic_extent)
          ? (__static_prev_ngroups * 2)
          : ::cuda::std::dynamic_extent;

      using _MappingResult =
        __mapping_result<__static_ngroups, ::cuda::std::dynamic_extent, _PrevMappingResult::is_always_exhaustive(), false>;

      if constexpr (!_PrevMappingResult::is_always_exhaustive())
      {
        if (!__prev_mapping_result.is_valid())
        {
          return _MappingResult::__invalid();
        }
      }

      const auto __pred      = ::cuda::experimental::coop::__extract_label(__fn_(__prev_mapping_result));
      const auto __prev_mask = __prev_mapping_result.lane_mask().value();

      auto __match_mask = ::__ballot_sync(__prev_mask, __pred);
      if (!__pred)
      {
        __match_mask = (~__match_mask) & __prev_mask;
      }
      return _MappingResult{
        __prev_mapping_result.group_count() * 2,
        __prev_mapping_result.group_rank() + ((__pred) ? __prev_mapping_result.group_count() : 0u),
        static_cast<uint32_t>(::cuda::std::popcount(__match_mask)),
        static_cast<uint32_t>(::cuda::std::popcount(__match_mask & ::cuda::ptx::get_sreg_lanemask_lt())),
        ::cuda::device::lane_mask{__match_mask}};
    }
    else
    {
      using _MappingResult =
        __mapping_result<::cuda::std::dynamic_extent,
                         ::cuda::std::dynamic_extent,
                         _PrevMappingResult::is_always_exhaustive(),
                         false>;

      uint32_t __match_mask{};
      if (__prev_mapping_result.is_valid())
      {
        const auto __label     = ::cuda::experimental::coop::__extract_label(__fn_(__prev_mapping_result));
        const auto __prev_mask = __prev_mapping_result.lane_mask().value();

        if constexpr (sizeof(__label) <= sizeof(uint64_t))
        {
          __match_mask = ::__match_any_sync(__prev_mask, __label);
        }
#  if _CCCL_HAS_INT128()
        else if constexpr (sizeof(__label) <= sizeof(__uint128_t))
        {
          __match_mask = ::__match_any_sync(__prev_mask, static_cast<uint64_t>(__label));
          __match_mask &= ::__match_any_sync(__prev_mask, static_cast<uint64_t>(__label >> 64));
        }
#  endif // _CCCL_HAS_INT128()
        else
        {
          static_assert(::cuda::std::__always_false_v<_Unit>, "Unsupported label type");
        }
      }

      const auto __unit_count = static_cast<uint32_t>(::cuda::std::popcount(__match_mask));
      const auto __unit_rank =
        static_cast<uint32_t>(::cuda::std::popcount(__match_mask & ::cuda::ptx::get_sreg_lanemask_lt()));

      const auto __parent_mask = __parent.__mapping_result().lane_mask().value();
      const auto __root_mask   = ::__ballot_sync(__parent_mask, __prev_mapping_result.is_valid() && __unit_rank == 0);
      const auto __root_lane   = ::cuda::std::countr_zero(__match_mask);

      if (!__prev_mapping_result.is_valid())
      {
        return _MappingResult::__invalid();
      }

      const auto __group_count = static_cast<uint32_t>(::cuda::std::popcount(__root_mask));
      const auto __group_rank =
        static_cast<uint32_t>(::cuda::std::popcount(::cuda::bit_trunc(__root_mask, __root_lane)));

      return _MappingResult{
        __group_count, __group_rank, __unit_count, __unit_rank, ::cuda::device::lane_mask{__match_mask}};
    }
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const non_exhaustive_t&,
      const _Unit& __unit,
      const _ParentGroup& __parent,
      const _PrevMappingResult& __prev_mapping_result)
  {
    return map(__unit, __parent, __prev_mapping_result);
  }
};

template <class _PredFn>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES labeled_partition(_PredFn) -> labeled_partition<_PredFn>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_LABELED_PARTITION_CUH
