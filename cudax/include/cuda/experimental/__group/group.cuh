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
template <class _Unit, class _Level, class _Mapping, class _Hierarchy, class _Synchronizer>
class group
{
  using _MappingResult        = __group_mapping_result_t<_Mapping, _Unit, _Level, _Hierarchy>;
  using _SynchronizerInstance = __group_synchronizer_instance_t<_Synchronizer, _Unit, _Level, _Mapping, _MappingResult>;

  static_assert(__group_mapping_result<_MappingResult>);

  _Hierarchy __hier_;
  _Mapping __mapping_;
  _MappingResult __mapping_result_;
  _Synchronizer __synchronizer_;
  _SynchronizerInstance __synchronizer_instance_;

public:
  using unit_type             = _Unit;
  using level_type            = _Level;
  using mapping_type          = _Mapping;
  using __mapping_result_type = _MappingResult;
  using hierarchy_type        = _Hierarchy;
  using synchronizer_type     = _Synchronizer;

  // todo(dabayer): Remove _Level and _HierarchyLike parameters and take a base group instead.
  _CCCL_TEMPLATE(class _HierarchyLike)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Hierarchy, __hierarchy_type_of<_HierarchyLike>>)
  _CCCL_DEVICE_API explicit group(
    const _Unit&,
    const _Level&,
    const _Mapping& __mapping,
    const _HierarchyLike& __hier_like,
    const _Synchronizer& __synchronizer) noexcept
      : __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
      , __mapping_{__mapping}
      , __mapping_result_{__mapping_.map(_Unit{}, _Level{}, ::cuda::__unpack_hierarchy_if_needed(__hier_like))}
      , __synchronizer_{__synchronizer}
      , __synchronizer_instance_{__synchronizer_.make_instance(_Unit{}, _Level{}, __mapping_, __mapping_result_)}
  {
    ::cuda::experimental::__check_mapping_result(__mapping_result_);
  }

  [[nodiscard]] _CCCL_DEVICE_API const _Hierarchy& hierarchy() const noexcept
  {
    return __hier_;
  }

  // todo(dabayer): Do we want to expose mapping getter?
  [[nodiscard]] _CCCL_DEVICE_API const _Mapping& mapping() const noexcept
  {
    return __mapping_;
  }

  // todo(dabayer): Do we want to expose mapping result getter?
  [[nodiscard]] _CCCL_DEVICE_API _MappingResult __mapping_result() const noexcept
  {
    return __mapping_result_;
  }

  // todo(dabayer): Do we want to expose synchronizer getter?
  [[nodiscard]] _CCCL_DEVICE_API const _Synchronizer& synchronizer() const noexcept
  {
    return __synchronizer_;
  }

  // todo(dabayer): Do we want to expose .arrive() and .wait()? Do we want to implement .sync() using them? Do we want
  //                aligned/unaligned variants?
  _CCCL_DEVICE_API void sync() noexcept
  {
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

_CCCL_TEMPLATE(class _Unit, class _Level, ::cuda::std::size_t _Np, class _HierarchyLike, class _Synchronizer)
_CCCL_REQUIRES(__is_hierarchy_level_v<_Unit> _CCCL_AND __is_hierarchy_level_v<_Level> _CCCL_AND
                 __is_or_has_hierarchy_member_v<_HierarchyLike>)
_CCCL_DEVICE group(const _Unit&, const _Level&, const group_by<_Np>&, const _HierarchyLike&, const _Synchronizer&)
  -> group<_Unit, _Level, group_by<_Np>, __hierarchy_type_of<_HierarchyLike>, _Synchronizer>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_GROUP_CUH
