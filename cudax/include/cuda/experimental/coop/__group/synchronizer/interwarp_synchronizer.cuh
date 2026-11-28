//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_INTERWARP_SYNCHRONIZER_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_INTERWARP_SYNCHRONIZER_CUH

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/hierarchy.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/range_to_sequence.cuh>
#include <cuda/experimental/coop/__group/synchronizer/arrival_token.cuh>

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED

namespace cuda::experimental::coop
{
using __interwarp_barrier_id _CCCL_NODEBUG                             = ::cuda::std::uint32_t;
inline constexpr __interwarp_barrier_id __dynamic_interwarp_barrier_id = ~0u;

template <__interwarp_barrier_id _BarrierId, ::cuda::std::size_t _ThreadCount, class _PackedBarrierIds = void>
class __interwarp_synchronizer_instance
{
  static_assert(_BarrierId == __dynamic_interwarp_barrier_id || _BarrierId < 16, "Invalid interwarp barrier id");

  __interwarp_barrier_id __barrier_id_;
  ::cuda::std::uint32_t __thread_count_;

  _CCCL_HIDE_FROM_ABI __interwarp_synchronizer_instance() = default;

public:
  using arrival_token _CCCL_NODEBUG             = __dummy_arrival_token<__interwarp_synchronizer_instance>;
  using __packed_barrier_ids_type _CCCL_NODEBUG = _PackedBarrierIds;

  [[nodiscard]] _CCCL_DEVICE_API constexpr static __interwarp_synchronizer_instance __invalid()
  {
    return __interwarp_synchronizer_instance{};
  }

  _CCCL_DEVICE_API constexpr __interwarp_synchronizer_instance(
    __interwarp_barrier_id __barrier_id, ::cuda::std::uint32_t __thread_count)
      : __barrier_id_{__barrier_id}
      , __thread_count_{__thread_count}
  {
    _CCCL_ASSERT(__barrier_id_ < 16u, "Invalid interwarp barrier id");
    if constexpr (_BarrierId != __dynamic_interwarp_barrier_id)
    {
      _CCCL_ASSERT(__barrier_id_ == _BarrierId, "Invalid interwarp barrier id");
    }

    if constexpr (_ThreadCount != ::cuda::std::dynamic_extent)
    {
      _CCCL_ASSERT(__thread_count == _ThreadCount, "Invalid thread count");
    }
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _Hierarchy&) const
  {
    __interwarp_barrier_id __barrier_id = __barrier_id_;
    if constexpr (_BarrierId != __dynamic_interwarp_barrier_id)
    {
      __barrier_id = _BarrierId;
    }

    ::cuda::std::uint32_t __thread_count = __thread_count_;
    if constexpr (_ThreadCount != ::cuda::std::dynamic_extent)
    {
      __thread_count = ::cuda::std::uint32_t{_ThreadCount};
    }

    ::__barrier_sync_count(__barrier_id, __thread_count);
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync_aligned(const _Hierarchy& __hier) const
  {
    // The .aligned version of the barrier instruction requires whole CTA to execute the same instruction, which is not
    // usable with groups. That means that we need to always use the unaligned version.
    do_sync(__hier);
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API arrival_token do_arrive(const _Hierarchy&) const
  {
    return arrival_token{};
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API arrival_token do_arrive_aligned(const _Hierarchy&) const
  {
    return arrival_token{};
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait(const _Hierarchy& __hier, [[maybe_unused]] arrival_token&& __tok) const
  {
    // Construct a temporary object, so we set __tok to moved-from state.
    _CCCL_ASSERT(arrival_token{::cuda::std::move(__tok)}.__is_valid(),
                 "invalid arrival token passed to do_wait function");
    do_sync(__hier);
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait_aligned(const _Hierarchy& __hier, [[maybe_unused]] arrival_token&& __tok) const
  {
    // Construct a temporary object, so we set __tok to moved-from state.
    _CCCL_ASSERT(arrival_token{::cuda::std::move(__tok)}.__is_valid(),
                 "invalid arrival token passed to do_wait_aligned function");
    do_sync_aligned(__hier);
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void deinit(const _MappingResult&, const _Hierarchy&)
  {}
};

template <class _Unit, class _ParentGroup, class _MappingResult, ::cuda::std::size_t _IdCount>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_CONSTEVAL ::cuda::std::size_t __interwarp_static_thread_count()
{
  static_assert(::cuda::std::is_same_v<_Unit, warp_level>
                  && ::cuda::std::is_same_v<typename _ParentGroup::boundary_level, block_level>,
                "interwarp_synchronizer can be only used to group warps within a single block");

  constexpr auto __group_count = _MappingResult::static_group_count();
  constexpr auto __group_rank  = _MappingResult::static_group_rank();

  static_assert(__group_count == ::cuda::std::dynamic_extent || __group_count <= _IdCount,
                "insufficient amount of warp barrier ids were passed");
  static_assert(__group_rank == ::cuda::std::dynamic_extent || __group_rank < _IdCount,
                "group rank exceeds the supplied warp barrier IDs");

  constexpr auto __warp_count = _MappingResult::static_unit_count();
  return __warp_count == ::cuda::std::dynamic_extent ? ::cuda::std::dynamic_extent : __warp_count * 32;
}

template <class _RangeView>
class interwarp_synchronizer
{
  static_assert(::cuda::std::ranges::__static_size_v<_RangeView> != 0, "at least one warp barrier ID is required");
  static_assert(::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_RangeView>, __interwarp_barrier_id>,
                "_RangeView's value_type must be convertible to the interwarp barrier id");

  _RangeView __barrier_ids_;

  template <class _Range, class _Unit, class _ParentGroup, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API static auto
  __make_instance(_Range& __barrier_ids, const _Unit&, const _ParentGroup&, const _MappingResult& __mapping_result)
  {
    constexpr auto __thread_count = ::cuda::experimental::coop::
      __interwarp_static_thread_count<_Unit, _ParentGroup, _MappingResult, ::cuda::std::ranges::__static_size_v<_Range>>();
    using _SynchronizerInstance = __interwarp_synchronizer_instance<__dynamic_interwarp_barrier_id, __thread_count>;

    // Default construct the synchronizer instance for units that don't have a valid mapping result.
    if (!__mapping_result.is_valid())
    {
      return _SynchronizerInstance::__invalid();
    }

    if constexpr (::cuda::std::ranges::sized_range<_Range>)
    {
      _CCCL_ASSERT(__mapping_result.group_count() <= ::cuda::std::ranges::size(__barrier_ids),
                   "insufficient amount of warp barrier ids were passed");

      auto __it = ::cuda::std::ranges::begin(__barrier_ids);
      ::cuda::std::ranges::advance(__it, __mapping_result.group_rank());
      return _SynchronizerInstance{static_cast<__interwarp_barrier_id>(*__it), __mapping_result.unit_count() * 32};
    }
    else
    {
      ::cuda::std::uint32_t __i = 0;
      auto __it                 = ::cuda::std::ranges::begin(__barrier_ids);
      do
      {
        if (__i > 0)
        {
          ++__it;
        }
        _CCCL_ASSERT(__it != ::cuda::std::ranges::end(__barrier_ids),
                     "insufficient amount of warp barrier ids were passed");
      } while (__i++ < __mapping_result.group_rank());
      return _SynchronizerInstance{static_cast<__interwarp_barrier_id>(*__it), __mapping_result.unit_count() * 32};
    }
  }

public:
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_RangeView, ::cuda::std::ranges::views::all_t<_Range>>)
  _CCCL_DEVICE_API interwarp_synchronizer(_Range&& __barrier_ids) // NOLINT(bugprone-forwarding-reference-overload)
      : __barrier_ids_{::cuda::std::ranges::views::all(::cuda::std::forward<_Range>(__barrier_ids))}
  {}

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _MappingResult)
  _CCCL_REQUIRES((!::cuda::std::ranges::range<const _RangeView>) )
  [[nodiscard]] _CCCL_DEVICE_API auto
  make_instance(const _Unit& __unit, const _ParentGroup& __parent, const _MappingResult& __mapping_result)
  {
    return __make_instance(__barrier_ids_, __unit, __parent, __mapping_result);
  }

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _MappingResult)
  _CCCL_REQUIRES(::cuda::std::ranges::range<const _RangeView>)
  [[nodiscard]] _CCCL_DEVICE_API auto
  make_instance(const _Unit& __unit, const _ParentGroup& __parent, const _MappingResult& __mapping_result) const
  {
    return __make_instance(__barrier_ids_, __unit, __parent, __mapping_result);
  }
};

template <::cuda::std::size_t... _BarrierIds>
class interwarp_synchronizer<::cuda::std::index_sequence<_BarrierIds...>>
{
  static_assert(sizeof...(_BarrierIds) > 0, "at least one warp barrier ID is required");
  static_assert(((_BarrierIds < 16) && ...), "Invalid native warp barrier id");
  static_assert(::cuda::std::popcount((0u | ... | (1u << _BarrierIds))) == sizeof...(_BarrierIds),
                "native warp barrier ids must be unique");

  using _PackedBarrierIdsStorage _CCCL_NODEBUG =
    ::cuda::std::conditional_t<sizeof...(_BarrierIds) <= 8, ::cuda::std::uint32_t, ::cuda::std::uint64_t>;

  [[nodiscard]] _CCCL_DEVICE_API static _CCCL_CONSTEVAL _PackedBarrierIdsStorage __make_packed_ids()
  {
    _PackedBarrierIdsStorage __ret = 0;
    ::cuda::std::size_t __shift    = 0;
    ((__ret |= _PackedBarrierIdsStorage{_BarrierIds} << __shift, __shift += 4), ...);
    return __ret;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr __interwarp_barrier_id __barrier_id_at(::cuda::std::uint32_t __rank)
  {
    _CCCL_ASSERT(__rank < sizeof...(_BarrierIds), "group rank exceeds the supplied warp barrier IDs");
    return static_cast<__interwarp_barrier_id>((__make_packed_ids() >> (4 * __rank)) & 0xfu);
  }

public:
  _CCCL_TEMPLATE(class _StaticRange)
  _CCCL_REQUIRES(
    ::cuda::std::is_same_v<__to_index_sequence_t<_StaticRange>, ::cuda::std::index_sequence<_BarrierIds...>>)
  _CCCL_DEVICE_API explicit constexpr interwarp_synchronizer(const _StaticRange&) {}

  template <class _Unit, class _ParentGroup, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  make_instance(const _Unit&, const _ParentGroup&, const _MappingResult& __mapping_result) const
  {
    constexpr auto __thread_count = ::cuda::experimental::coop::
      __interwarp_static_thread_count<_Unit, _ParentGroup, _MappingResult, sizeof...(_BarrierIds)>();
    constexpr auto __rank = _MappingResult::static_group_rank();
    constexpr auto __id =
      __rank == ::cuda::std::dynamic_extent
        ? __dynamic_interwarp_barrier_id
        : __barrier_id_at(static_cast<::cuda::std::uint32_t>(__rank));
    using _SynchronizerInstance =
      __interwarp_synchronizer_instance<__id,
                                        __thread_count,
                                        ::cuda::std::integral_constant<_PackedBarrierIdsStorage, __make_packed_ids()>>;

    if (!__mapping_result.is_valid())
    {
      return _SynchronizerInstance::__invalid();
    }

    _CCCL_ASSERT(__mapping_result.group_count() <= sizeof...(_BarrierIds),
                 "insufficient amount of warp barrier ids were passed");
    const auto __barrier_id =
      __rank == ::cuda::std::dynamic_extent ? __barrier_id_at(__mapping_result.group_rank()) : __id;
    return _SynchronizerInstance{__barrier_id, __mapping_result.unit_count() * 32};
  }
};

_CCCL_TEMPLATE(class _Range)
_CCCL_REQUIRES(
  ::cuda::std::ranges::viewable_range<_Range>
    _CCCL_AND ::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_Range>, __interwarp_barrier_id>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES interwarp_synchronizer(_Range&&)
  -> interwarp_synchronizer<::cuda::std::ranges::views::all_t<_Range>>;

template <class _StaticRange>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES interwarp_synchronizer(const _StaticRange&)
  -> interwarp_synchronizer<__to_index_sequence_t<_StaticRange>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_INTERWARP_SYNCHRONIZER_CUH
