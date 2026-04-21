//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_GROUP_CUH
#define _CUDA_EXPERIMENTAL___GROUP_GROUP_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__bit/bitmask.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/barrier>
#include <cuda/hierarchy>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/span>

#include <cuda/experimental/__group/concepts.cuh>
#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/mapping/group_by.cuh>
#include <cuda/experimental/__group/this_group.cuh>
#include <cuda/experimental/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _Unit, class _ParentGroup, class _Mapping, class _Synchronizer>
class group
{
  static_assert(__is_hierarchy_level_v<_Unit>);
  static_assert(is_group<_ParentGroup>);

  // todo(dabayer): Allow groups stacking and remove this.
  static_assert(__is_this_group_v<_ParentGroup>);

  // todo(dabayer): static_assert that _Unit is (under) typename _ParentGroup::unit_type

  using _ParentMappingResult = typename _ParentGroup::__mapping_result_type;
  using _MappingResult       = __group_mapping_result_t<_Mapping, _Unit, _ParentGroup>;
  using _SynchronizerInstance =
    __group_synchronizer_instance_t<_Synchronizer, _Unit, _ParentGroup, _Mapping, _MappingResult>;
  static_assert(__group_mapping_result<_MappingResult>);

  typename _ParentGroup::hierarchy_type __hier_;
  _Mapping __mapping_;
  _MappingResult __mapping_result_;
  _Synchronizer __synchronizer_;
  _SynchronizerInstance __synchronizer_instance_;

  [[nodiscard]] _CCCL_DEVICE_API static _MappingResult
  __do_mapping(const _Mapping& __mapping, const _ParentGroup& __parent) noexcept
  {
    // Do not invoke the mapping for threads that are not part of the parent group.
    if constexpr (!_ParentMappingResult::is_always_exhaustive())
    {
      if (!__parent.__mapping_result().is_valid())
      {
        return _MappingResult::invalid();
      }
    }

    const auto __mapping_result = __mapping.map(_Unit{}, __parent);
    if (__mapping_result.is_valid())
    {
      _CCCL_ASSERT(__mapping_result.group_rank() < __mapping_result.group_count(), "invalid group rank");
      _CCCL_ASSERT(__mapping_result.rank() < __mapping_result.count(), "invalid rank");
    }
    return __mapping_result;
  }

  [[nodiscard]] _CCCL_DEVICE_API static _SynchronizerInstance __make_synchronizer_instance(
    const _Synchronizer& __synchronizer,
    const _ParentGroup& __parent,
    const _Mapping& __mapping,
    const _MappingResult& __mapping_result) noexcept
  {
    // Do not invoke the synchronizer instance creation for threads that are not part of the parent group. On the other
    // hand threads that are not part of this group must create the synchronizer instance, too, because the operation
    // can synchronize the parent group.
    if constexpr (!_ParentMappingResult::is_always_exhaustive())
    {
      if (!__parent.__mapping_result().is_valid())
      {
        return _MappingResult::invalid();
      }
    }
    return __synchronizer.make_instance(_Unit{}, __parent, __mapping, __mapping_result);
  }

public:
  using unit_type             = _Unit;
  using level_type            = typename _ParentGroup::level_type;
  using hierarchy_type        = typename _ParentGroup::hierarchy_type;
  using mapping_type          = _Mapping;
  using __mapping_result_type = _MappingResult;
  using synchronizer_type     = _Synchronizer;

  _CCCL_DEVICE_API explicit group(
    const _Unit& __unit,
    const _ParentGroup& __parent,
    const _Mapping& __mapping,
    const _Synchronizer& __synchronizer) noexcept
      : __hier_{__parent.hierarchy()}
      , __mapping_{__mapping}
      , __mapping_result_{__do_mapping(__mapping_, __parent)}
      , __synchronizer_{__synchronizer}
      , __synchronizer_instance_{__make_synchronizer_instance(__synchronizer_, __parent, __mapping_, __mapping_result_)}
  {}

  [[nodiscard]] _CCCL_DEVICE_API const hierarchy_type& hierarchy() const noexcept
  {
    return __hier_;
  }

  // todo(dabayer): Do we want to expose mapping getter?
  [[nodiscard]] _CCCL_DEVICE_API const mapping_type& mapping() const noexcept
  {
    return __mapping_;
  }

  // todo(dabayer): Do we want to expose mapping result getter?
  [[nodiscard]] _CCCL_DEVICE_API _MappingResult __mapping_result() const noexcept
  {
    return __mapping_result_;
  }

  // todo(dabayer): Do we want to expose synchronizer getter?
  [[nodiscard]] _CCCL_DEVICE_API const synchronizer_type& synchronizer() const noexcept
  {
    return __synchronizer_;
  }

  // todo(dabayer): Do we want to expose .arrive() and .wait()? Do we want to implement .sync() using them? Do we want
  //                aligned/unaligned variants?
  _CCCL_DEVICE_API void sync() noexcept
  {
    // Skip the synchronization for threads that are not part of this group.
    if constexpr (!_MappingResult::is_always_exhaustive())
    {
      if (!__mapping_result_.is_valid())
      {
        return;
      }
    }
    __synchronizer_instance_.do_sync(__mapping_result_, __synchronizer_);
  }

  _CCCL_DEVICE_API void sync_aligned() noexcept
  {
    // Skip the synchronization for threads that are not part of this group.
    if constexpr (!_MappingResult::is_always_exhaustive())
    {
      if (!__mapping_result_.is_valid())
      {
        return;
      }
    }
    __synchronizer_instance_.do_sync_aligned(__mapping_result_, __synchronizer_);
  }
};

_CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _Mapping, class _Synchronizer)
_CCCL_REQUIRES(__is_hierarchy_level_v<_Unit> _CCCL_AND is_group<_ParentGroup>)
_CCCL_DEVICE group(const _Unit&, const _ParentGroup&, const _Mapping&, const _Synchronizer&)
  -> group<_Unit, _ParentGroup, _Mapping, _Synchronizer>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_GROUP_CUH
