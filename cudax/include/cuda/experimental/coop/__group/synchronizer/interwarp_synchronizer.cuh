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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/concepts.cuh>
#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/synchronizer/arrival_token.cuh>
#include <cuda/experimental/coop/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
using __interwarp_barrier_id = ::cuda::std::uint32_t;

template <::cuda::std::size_t _ThreadCount>
class __interwarp_synchronizer_instance
{
  __interwarp_barrier_id __barrier_id_;

public:
  using arrival_token _CCCL_NODEBUG = __dummy_arrival_token<__interwarp_synchronizer_instance>;

  _CCCL_DEVICE_API
  __interwarp_synchronizer_instance(__interwarp_barrier_id __barrier_id, ::cuda::std::uint32_t __thread_count)
      : __barrier_id_{__barrier_id}
  {
    _CCCL_ASSERT(__barrier_id_ < 16u, "Invalid native warp barrier id");
    _CCCL_ASSERT(__thread_count == 0 || __thread_count == ::cuda::std::uint32_t{_ThreadCount}, "Invalid thread count");
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _Hierarchy&) const
  {
    ::__barrier_sync_count(__barrier_id_, ::cuda::std::uint32_t{_ThreadCount});
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync_aligned(const _Hierarchy&) const
  {
    // The .aligned version of the barrier instruction requires whole CTA to execute the same instruction, which is not
    // usable with groups. That means that we need to always use the unaligned version.
    ::__barrier_sync_count(__barrier_id_, ::cuda::std::uint32_t{_ThreadCount});
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
    _CCCL_ASSERT(arrival_token{::cuda::std::move(__tok)}.__valid(), "invalid arrival token passed to do_wait function");
    do_sync(__hier);
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait_aligned(const _Hierarchy& __hier, [[maybe_unused]] arrival_token&& __tok) const
  {
    // Construct a temporary object, so we set __tok to moved-from state.
    _CCCL_ASSERT(arrival_token{::cuda::std::move(__tok)}.__valid(),
                 "invalid arrival token passed to do_wait_aligned function");
    do_sync_aligned(__hier);
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void deinit(const _MappingResult&, const _Hierarchy&)
  {}
};

template <>
class __interwarp_synchronizer_instance<::cuda::std::dynamic_extent>
{
  __interwarp_barrier_id __barrier_id_;
  ::cuda::std::uint32_t __thread_count_;

public:
  using arrival_token _CCCL_NODEBUG = __dummy_arrival_token<__interwarp_synchronizer_instance>;

  _CCCL_DEVICE_API
  __interwarp_synchronizer_instance(__interwarp_barrier_id __barrier_id, ::cuda::std::uint32_t __thread_count)
      : __barrier_id_{__barrier_id}
      , __thread_count_{__thread_count}
  {
    _CCCL_ASSERT(__barrier_id_ < 16u, "Invalid native warp barrier id");
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _Hierarchy&) const
  {
    ::__barrier_sync_count(__barrier_id_, __thread_count_);
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync_aligned(const _Hierarchy&) const
  {
    // The .aligned version of the barrier instruction requires whole CTA to execute the same instruction, which is not
    // usable with groups. That means that we need to always use the unaligned version.
    ::__barrier_sync_count(__barrier_id_, __thread_count_);
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
    _CCCL_ASSERT(arrival_token{::cuda::std::move(__tok)}.__valid(), "invalid arrival token passed to do_wait function");
    do_sync(__hier);
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait_aligned(const _Hierarchy& __hier, [[maybe_unused]] arrival_token&& __tok) const
  {
    // Construct a temporary object, so we set __tok to moved-from state.
    _CCCL_ASSERT(arrival_token{::cuda::std::move(__tok)}.__valid(),
                 "invalid arrival token passed to do_wait_aligned function");
    do_sync_aligned(__hier);
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void deinit(const _MappingResult&, const _Hierarchy&)
  {}
};

template <class _RangeView>
class interwarp_synchronizer
{
  static_assert(::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_RangeView>, __interwarp_barrier_id>,
                "_RangeView's value_type must be convertible to the interwarp barrier id");

  _RangeView __barrier_ids_;

public:
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_RangeView, ::cuda::std::ranges::views::all_t<_Range>>)
  _CCCL_DEVICE_API interwarp_synchronizer(_Range&& __barrier_ids)
      : __barrier_ids_{::cuda::std::ranges::views::all(::cuda::std::forward<_Range>(__barrier_ids))}
  {}

  template <class _Unit, class _ParentGroup, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  make_instance(const _Unit&, const _ParentGroup&, const _MappingResult& __mapping_result)
  {
    using _Level = typename _ParentGroup::level_type;

    constexpr auto __static_group_count      = _MappingResult::static_group_count();
    constexpr auto __static_barrier_id_count = ::cuda::std::ranges::__static_size_v<_RangeView>;
    if constexpr (__static_group_count != ::cuda::std::dynamic_extent
                  && __static_barrier_id_count != ::cuda::std::dynamic_extent)
    {
      static_assert(__static_group_count <= __static_barrier_id_count,
                    "insufficient amount of warp barrier ids were passed");
    }

    static_assert(::cuda::std::is_same_v<_Unit, warp_level> && ::cuda::std::is_same_v<_Level, block_level>,
                  "interwarp_synchronizer can be only used to group warps within a single block");

    constexpr auto __static_unit_count = _MappingResult::static_unit_count();
    using _SynchronizerInstance        = __interwarp_synchronizer_instance<
      (__static_unit_count != ::cuda::std::dynamic_extent) ? __static_unit_count * 32 : ::cuda::std::dynamic_extent>;

    // Default construct the synchronizer instance for units that don't have a valid mapping result.
    if (!__mapping_result.is_valid())
    {
      return _SynchronizerInstance{0, 0};
    }

    if constexpr (::cuda::std::ranges::sized_range<_RangeView>)
    {
      _CCCL_ASSERT(__mapping_result.group_count() <= ::cuda::std::ranges::size(__barrier_ids_),
                   "insufficient amount of warp barrier ids were passed");

      auto __it = ::cuda::std::ranges::begin(__barrier_ids_);
      ::cuda::std::ranges::advance(__it, __mapping_result.group_rank());
      return _SynchronizerInstance{static_cast<__interwarp_barrier_id>(*__it), __mapping_result.unit_count() * 32};
    }
    else
    {
      ::cuda::std::uint32_t __i = 0;
      auto __it                 = ::cuda::std::ranges::begin(__barrier_ids_);
      do
      {
        if (__i > 0)
        {
          ++__it;
        }
        _CCCL_ASSERT(__it != ::cuda::std::ranges::end(__barrier_ids_),
                     "insufficient amount of warp barrier ids were passed");
      } while (__i++ < __mapping_result.group_rank());
      return _SynchronizerInstance{static_cast<__interwarp_barrier_id>(*__it), __mapping_result.unit_count() * 32};
    }
  }
};

_CCCL_TEMPLATE(class _Range)
_CCCL_REQUIRES(
  ::cuda::std::ranges::viewable_range<_Range>
    _CCCL_AND ::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_Range>, __interwarp_barrier_id>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES interwarp_synchronizer(_Range&&)
  -> interwarp_synchronizer<::cuda::std::ranges::views::all_t<_Range>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_INTERWARP_SYNCHRONIZER_CUH
