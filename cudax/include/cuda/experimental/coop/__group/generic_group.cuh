//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_GENERIC_GROUP_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_GENERIC_GROUP_CUH

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
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/coop/__group/concepts.cuh>
#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/group_view.cuh>
#include <cuda/experimental/coop/__group/mapping/common.cuh>
#include <cuda/experimental/coop/__group/virtual_group.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <class _Unit, class _ParentGroup, class _MappingResult, class _Synchronizer>
using __group_synchronizer_instance_t = decltype(::cuda::std::declval<_Synchronizer&>().make_instance(
  ::cuda::std::declval<const _Unit&>(),
  ::cuda::std::declval<const _ParentGroup&>(),
  ::cuda::std::declval<const _MappingResult&>()));

template <class _Unit, class _ParentGroup, class _MappingResult, class _SynchronizerInstance>
class generic_group
{
  static_assert(__is_hierarchy_level_v<_Unit>);
  static_assert(group<_ParentGroup>);
  static_assert(__unit_same_as_or_below_v<_Unit, typename _ParentGroup::unit_type>,
                "unit_type must be same as or below _ParentGroup's unit_type");

  using _Hierarchy _CCCL_NODEBUG           = typename _ParentGroup::hierarchy_type;
  using _ParentMappingResult _CCCL_NODEBUG = typename _ParentGroup::__mapping_result_type;
  static_assert(__group_mapping_result<_MappingResult>);

  _Hierarchy __hier_;
  _MappingResult __mapping_result_;
  _SynchronizerInstance __synchronizer_instance_;

public:
  using unit_type                    = _Unit;
  using level_type                   = typename _ParentGroup::level_type;
  using hierarchy_type               = _Hierarchy;
  using __mapping_result_type        = _MappingResult;
  using __synchronizer_instance_type = _SynchronizerInstance;
  using arrival_token                = typename _SynchronizerInstance::arrival_token;

  _CCCL_TEMPLATE(class _Mapping, class _Synchronizer)
  _CCCL_REQUIRES(
    ::cuda::std::is_same_v<_MappingResult, __group_mapping_result_t<_Unit, _ParentGroup, _Mapping>> _CCCL_AND ::cuda::
      std::is_same_v<_SynchronizerInstance,
                     __group_synchronizer_instance_t<_Unit, _ParentGroup, _MappingResult, _Synchronizer>>)
  _CCCL_DEVICE_API explicit generic_group(
    const _Unit& __unit, const _ParentGroup& __parent, _Mapping&& __mapping, _Synchronizer&& __synchronizer)
      : __hier_{__parent.hierarchy()}
      , __mapping_result_{::cuda::experimental::coop::__do_group_mapping(
          __unit, __parent, ::cuda::std::forward<_Mapping>(__mapping))}
      , __synchronizer_instance_{
          ::cuda::std::forward<_Synchronizer>(__synchronizer).make_instance(__unit, __parent, __mapping_result_)}
  {}

  _CCCL_TEMPLATE(class _ParentUnit, class _ParentMappingResult, class _Mapping, class _Synchronizer)
  _CCCL_REQUIRES(
    ::cuda::std::is_same_v<
      _MappingResult,
      __group_mapping_result_t<_Unit, group_view<_ParentUnit, _ParentGroup, _ParentMappingResult>, _Mapping>>
      _CCCL_AND ::cuda::std::is_same_v<
        _SynchronizerInstance,
        __group_synchronizer_instance_t<_Unit,
                                        group_view<_ParentUnit, _ParentGroup, _ParentMappingResult>,
                                        _MappingResult,
                                        _Synchronizer>>)
  _CCCL_DEVICE_API explicit generic_group(
    const _Unit& __unit,
    const group_view<_ParentUnit, _ParentGroup, _ParentMappingResult>& __parent,
    _Mapping&& __mapping,
    _Synchronizer&& __synchronizer)
      : __hier_{__parent.hierarchy()}
      , __mapping_result_{::cuda::experimental::coop::__do_group_mapping(
          __unit, __parent, ::cuda::std::forward<_Mapping>(__mapping))}
      , __synchronizer_instance_{
          ::cuda::std::forward<_Synchronizer>(__synchronizer).make_instance(__unit, __parent, __mapping_result_)}
  {}

  // internal constructor used by make_group(non_exhaustive_t, ...)
  _CCCL_DEVICE_API explicit generic_group(
    const non_exhaustive_t&,
    const _Unit& __unit,
    const _ParentGroup& __parent,
    const ::cuda::std::type_identity_t<_MappingResult>& __mapping_result,
    const ::cuda::std::type_identity_t<_SynchronizerInstance>& __synchronizer_instance)
      : __hier_{__parent.hierarchy()}
      , __mapping_result_{__mapping_result}
      , __synchronizer_instance_{__synchronizer_instance}
  {}

  template <class _ParentUnit, class _ParentMappingResult>
  _CCCL_DEVICE_API explicit generic_group(
    const non_exhaustive_t&,
    const _Unit& __unit,
    const group_view<_ParentUnit, _ParentGroup, _ParentMappingResult>& __parent,
    const ::cuda::std::type_identity_t<_MappingResult>& __mapping_result,
    const ::cuda::std::type_identity_t<_SynchronizerInstance>& __synchronizer_instance)
      : __hier_{__parent.hierarchy()}
      , __mapping_result_{__mapping_result}
      , __synchronizer_instance_{__synchronizer_instance}
  {}

  // Groups can't be copied, moved nor assigned.
  generic_group(const generic_group&)            = delete;
  generic_group(generic_group&&)                 = delete;
  generic_group& operator=(const generic_group&) = delete;
  generic_group& operator=(generic_group&&)      = delete;

  _CCCL_DEVICE_API ~generic_group()
  {
    __synchronizer_instance_.deinit(__mapping_result_, __hier_);
  }

  //! @brief Invokes the callable with a view specialized for this group's static unit count.
  //! Counts that are only known at runtime remain dynamic.
  template <class _Fn>
  _CCCL_DEVICE_API decltype(auto) with_static_unit_count(_Fn&& __fn) const
  {
    return ::cuda::experimental::coop::__with_static_unit_count(*this, ::cuda::std::forward<_Fn>(__fn));
  }

  [[nodiscard]] _CCCL_DEVICE_API const hierarchy_type& hierarchy() const
  {
    return __hier_;
  }

  // todo(dabayer): Do we want to expose mapping result getter?
  [[nodiscard]] _CCCL_DEVICE_API _MappingResult __mapping_result() const
  {
    return __mapping_result_;
  }

  // todo(dabayer): Do we want to expose synchronizer instance getter?
  [[nodiscard]] _CCCL_DEVICE_API const _SynchronizerInstance& __synchronizer_instance() const
  {
    return __synchronizer_instance_;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous()
  {
    return _MappingResult::is_always_contiguous();
  }

  // todo(dabayer): Do we want to expose .arrive() and .wait()? Do we want to implement .sync() using them? Do we want
  //                aligned/unaligned variants?
  _CCCL_DEVICE_API void sync() const
  {
    __synchronizer_instance_.do_sync(__hier_);
  }

  _CCCL_DEVICE_API void sync_aligned() const
  {
    __synchronizer_instance_.do_sync_aligned(__hier_);
  }

  [[nodiscard]] _CCCL_DEVICE_API arrival_token arrive() const
  {
    return __synchronizer_instance_.do_arrive(__hier_);
  }

  [[nodiscard]] _CCCL_DEVICE_API arrival_token arrive_aligned() const
  {
    return __synchronizer_instance_.do_arrive_aligned(__hier_);
  }

  _CCCL_DEVICE_API void wait(arrival_token&& __tok) const
  {
    __synchronizer_instance_.do_wait(__hier_, ::cuda::std::move(__tok));
  }

  _CCCL_DEVICE_API void wait_aligned(arrival_token&& __tok) const
  {
    __synchronizer_instance_.do_wait_aligned(__hier_, ::cuda::std::move(__tok));
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count(const _ParentGroup&)
  {
    return _MappingResult::static_group_count();
  }

  template <class _Tp, class _QueryMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr _Tp
  __count_as_impl(const _QueryMappingResult& __mapping_result, const _Hierarchy&, const _ParentGroup&)
  {
    return static_cast<_Tp>(__mapping_result.group_count());
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp count_as(const _ParentGroup& __parent) const
  {
    return __count_as_impl<_Tp>(__mapping_result_, __hier_, __parent);
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr auto count(const _ParentGroup& __parent) const
  {
    return __count_as_impl<typename level_type::__product_type>(__mapping_result_, __hier_, __parent);
  }

  //! @brief Returns the group rank within the parent known at compile time, or @c cuda::std::dynamic_extent.
  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_rank(const _ParentGroup&)
  {
    return _MappingResult::static_group_rank();
  }

  template <class _Tp, class _QueryMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr _Tp
  __rank_as_impl(const _QueryMappingResult& __mapping_result, const _Hierarchy&, const _ParentGroup&)
  {
    return static_cast<_Tp>(__mapping_result.group_rank());
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API _Tp rank_as(const _ParentGroup& __parent) const
  {
    return __rank_as_impl<_Tp>(__mapping_result_, __hier_, __parent);
  }

  [[nodiscard]] _CCCL_DEVICE_API auto rank(const _ParentGroup& __parent) const
  {
    return __rank_as_impl<typename level_type::__product_type>(__mapping_result_, __hier_, __parent);
  }
};

_CCCL_TEMPLATE(
  class _Unit,
  class _ParentGroup,
  class _Mapping,
  class _Synchronizer,
  class _MappingResult        = __group_mapping_result_t<_Unit, _ParentGroup, _Mapping>,
  class _SynchronizerInstance = __group_synchronizer_instance_t<_Unit, _ParentGroup, _MappingResult, _Synchronizer>)
_CCCL_REQUIRES(__is_hierarchy_level_v<_Unit> _CCCL_AND group<_ParentGroup> _CCCL_AND
                 __unit_same_as_or_below_v<_Unit, typename _ParentGroup::unit_type>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES generic_group(const _Unit&, const _ParentGroup&, _Mapping&&, _Synchronizer&&)
  -> generic_group<_Unit, _ParentGroup, _MappingResult, _SynchronizerInstance>;

_CCCL_TEMPLATE(
  class _Unit,
  class _ParentUnit,
  class _ParentGroup,
  class _ParentMappingResult,
  class _Mapping,
  class _Synchronizer,
  class _ParentView           = group_view<_ParentUnit, _ParentGroup, _ParentMappingResult>,
  class _MappingResult        = __group_mapping_result_t<_Unit, _ParentView, _Mapping>,
  class _SynchronizerInstance = __group_synchronizer_instance_t<_Unit, _ParentView, _MappingResult, _Synchronizer>)
_CCCL_REQUIRES(__is_hierarchy_level_v<_Unit> _CCCL_AND __unit_same_as_or_below_v<_Unit, _ParentUnit>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES generic_group(
  const _Unit&, const group_view<_ParentUnit, _ParentGroup, _ParentMappingResult>&, _Mapping&&, _Synchronizer&&)
  -> generic_group<_Unit, _ParentGroup, _MappingResult, _SynchronizerInstance>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_GENERIC_GROUP_CUH
