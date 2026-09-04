//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_INTERWARP_SYNCHRONIZER_CUH
#define _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_INTERWARP_SYNCHRONIZER_CUH

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
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__group/concepts.cuh>
#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
using __interwarp_barrier_id = ::cuda::std::uint32_t;

class __interwarp_synchronizer_instance
{
  mutable __interwarp_barrier_id __barrier_id_;

public:
  [[nodiscard]] _CCCL_DEVICE_API static __interwarp_synchronizer_instance invalid() noexcept
  {
    __interwarp_synchronizer_instance __ret{0};
    __ret.__barrier_id_ = 0xffff'ffff;
    return __ret;
  }

  _CCCL_DEVICE_API __interwarp_synchronizer_instance(__interwarp_barrier_id __barrier_id) noexcept
      : __barrier_id_{__barrier_id}
  {
    _CCCL_ASSERT(__barrier_id_ < 16, "Invalid native warp barrier id");
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _MappingResult& __mapping_result, const _Hierarchy&) const noexcept
  {
    ::__barrier_sync_count(__barrier_id_, __mapping_result.unit_count() * 32);
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void do_sync_aligned(const _MappingResult& __mapping_result, const _Hierarchy&) const noexcept
  {
    // The .aligned version of the barrier instruction requires whole CTA to execute the same instruction, which is not
    // usable with groups. That means that we need to always use the unaligned version.
    ::__barrier_sync_count(__barrier_id_, __mapping_result.unit_count() * 32);
  }

  [[nodiscard]] _CCCL_DEVICE_API __interwarp_synchronizer_instance view() const noexcept
  {
    return *this;
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void deinit(const _MappingResult&, const _Hierarchy&) noexcept
  {}
};

template <class _Range>
class interwarp_synchronizer
{
  static_assert(::cuda::std::ranges::sized_range<_Range>, "_Range must be a sized range");
  static_assert(::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_Range>, __interwarp_barrier_id>,
                "_Range's value_type must be convertible to the interwarp barrier id");

  _Range __barrier_ids_;

public:
  using __synchronizer_instance = __interwarp_synchronizer_instance;

  _CCCL_TEMPLATE(class _Range2)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Range, _Range2>)
  _CCCL_DEVICE_API
  interwarp_synchronizer(_Range2&& __barrier_ids) noexcept(::cuda::std::is_nothrow_constructible_v<_Range, _Range2>)
      : __barrier_ids_{::cuda::std::forward<_Range2>(__barrier_ids)}
  {}

  template <class _Unit, class _ParentGroup, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API __synchronizer_instance
  make_instance(const _Unit&, const _ParentGroup&, const _MappingResult& __mapping_result) const noexcept
  {
    using _Level = typename _ParentGroup::level_type;

    static_assert(::cuda::std::is_same_v<_Unit, warp_level> && ::cuda::std::is_same_v<_Level, block_level>,
                  "interwarp_synchronizer can be only used to group warps within a single block");

    if (!__mapping_result.is_valid())
    {
      return __synchronizer_instance::invalid();
    }

    _CCCL_ASSERT(__mapping_result.group_count() <= 16,
                 "interwarp_synchronizer can't be used to synchronize more than 16 groups");

    _CCCL_ASSERT(__mapping_result.group_count() <= ::cuda::std::ranges::size(__barrier_ids_),
                 "insufficient amount of warp barrier ids were passed");

    auto __it = ::cuda::std::ranges::begin(__barrier_ids_);
    ::cuda::std::ranges::advance(__it, __mapping_result.group_rank());
    return __synchronizer_instance{static_cast<__interwarp_barrier_id>(*__it)};
  }
};

_CCCL_TEMPLATE(class _Range)
_CCCL_REQUIRES(
  ::cuda::std::ranges::sized_range<_Range>
    _CCCL_AND ::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_Range>, __interwarp_barrier_id>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES interwarp_synchronizer(_Range&&) -> interwarp_synchronizer<_Range>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_INTERWARP_SYNCHRONIZER_CUH
