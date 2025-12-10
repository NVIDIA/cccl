//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINER_MDARRAY_BASE_CUH
#define __CUDAX__CONTAINER_MDARRAY_BASE_CUH

#include <cuda/std/detail/__config>

#include <cuda/__cccl_config>
#include <cuda/__driver/driver_api.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/dispatch_copy_mdspan.cuh>

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/array>
#include <cuda/std/span>

#include <cuda/experimental/__container/mdarray_utils.cuh>

#include <cuda/std/__cccl/prologue.h>

// see also https://github.com/rapidsai/raft/blob/main/cpp/include/raft/core/mdarray.hpp
namespace cuda::experimental
{
template <typename _ElementType, typename _Extents, typename _LayoutPolicy>
struct __mdarray_constraints
{
  using extents_type = _Extents;
  using layout_type  = _LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using index_type   = typename extents_type::index_type;

  static constexpr bool __can_default_construct =
    (_Extents::rank_dynamic() > 0) && ::cuda::std::is_default_constructible_v<mapping_type>;

  template <class... _OtherIndexTypes>
  static constexpr bool __can_construct_from_handle_and_variadic =
    (::cuda::std::__mdspan_detail::__matches_dynamic_rank<extents_type, sizeof...(_OtherIndexTypes)>
     || ::cuda::std::__mdspan_detail::__matches_static_rank<extents_type, sizeof...(_OtherIndexTypes)>)
    && ::cuda::std::__mdspan_detail::__all_convertible_to_index_type<index_type, _OtherIndexTypes...>
    && ::cuda::std::is_constructible_v<mapping_type, extents_type>;

  template <class _OtherIndexType>
  static constexpr bool __is_constructible_from_index_type =
    ::cuda::std::is_convertible_v<const _OtherIndexType&, index_type>
    && ::cuda::std::is_nothrow_constructible_v<index_type, const _OtherIndexType&>
    && ::cuda::std::is_constructible_v<mapping_type, extents_type>;

  template <typename _OtherExtents, typename _OtherLayoutPolicy, typename _OtherAccessor>
  static constexpr bool __is_convertible_from =
    ::cuda::std::is_constructible_v<mapping_type, const typename _OtherLayoutPolicy::template mapping<_OtherExtents>&>;

  template <typename _OtherExtents, typename _OtherLayoutPolicy, typename _OtherAccessor>
  static constexpr bool __is_implicit_convertible_from =
    ::cuda::std::is_convertible_v<const typename _OtherLayoutPolicy::template mapping<_OtherExtents>&, mapping_type>;
};

// __mdarray_allocator_wrapper allows to initialize the allocator before allocating the memory
template <typename _Allocator>
struct __mdarray_allocator_wrapper
{
  _Allocator __allocator_;

  _CCCL_HIDE_FROM_ABI __mdarray_allocator_wrapper() = default;

  _CCCL_HOST_API explicit __mdarray_allocator_wrapper(const _Allocator& __allocator)
      : __allocator_{__allocator}
  {}

  [[nodiscard]] _CCCL_HOST_API _Allocator& __get_allocator() noexcept
  {
    return __allocator_;
  }

  [[nodiscard]] _CCCL_HOST_API const _Allocator& __get_allocator() const noexcept
  {
    return __allocator_;
  }
};

struct no_init_t
{};

template <typename _Derived,
          typename _Allocator,
          template <typename, typename, typename> class _ViewType,
          typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy>
class __base_mdarray
    : private __mdarray_allocator_wrapper<_Allocator>
    , public _ViewType<_ElementType, _Extents, _LayoutPolicy>
{
public:
  using mdspan_type    = _ViewType<_ElementType, _Extents, _LayoutPolicy>;
  using allocator_type = _Allocator;
  using extents_type   = _Extents;
  using mapping_type   = _LayoutPolicy::template mapping<extents_type>;
  using layout_type    = _LayoutPolicy;
  using element_type   = _ElementType;
  using pointer        = element_type*;

  using view_type       = _ViewType<_ElementType, _Extents, _LayoutPolicy>;
  using const_view_type = _ViewType<const _ElementType, _Extents, _LayoutPolicy>;

private:
  using __allocator_base = __mdarray_allocator_wrapper<allocator_type>;

  _CCCL_HOST_API void __release_storage() noexcept
  {
    if (this->data_handle() != nullptr)
    {
      const auto __size = ::cuda::experimental::__mapping_size_bytes(this->view());
      (this->__get_allocator().deallocate_sync(this->data_handle(), __size));
      static_cast<mdspan_type&>(*this) = mdspan_type{nullptr, this->mapping()};
    }
  }

  using __constraints = __mdarray_constraints<element_type, extents_type, layout_type>;

public:
  _CCCL_TEMPLATE(typename _Extents2 = extents_type)
  _CCCL_REQUIRES(__constraints::__can_default_construct)
  _CCCL_HOST_API __base_mdarray()
      : __base_mdarray{mapping_type{}, _Derived::__get_default_allocator()}
  {}

  _CCCL_HOST_API __base_mdarray(mapping_type __mapping)
      : __base_mdarray{__mapping, _Derived::__get_default_allocator()}
  {}

  _CCCL_HOST_API __base_mdarray(mapping_type __mapping, allocator_type __allocator)
      : __allocator_base{__allocator}
      , mdspan_type{
          static_cast<pointer>(__allocator.allocate_sync(__mapping.required_span_size() * sizeof(element_type))),
          __mapping}
  {
    static_cast<_Derived&>(*this).__init();
  }

  _CCCL_TEMPLATE(typename _Mapping2 = mapping_type)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Mapping2, const extents_type&>)
  _CCCL_HOST_API __base_mdarray(extents_type __ext)
      : __base_mdarray{mapping_type{__ext}, _Derived::__get_default_allocator()}
  {}

  _CCCL_TEMPLATE(typename _Mapping2 = mapping_type)
  _CCCL_REQUIRES(::cuda::std::is_constructible_v<_Mapping2, const extents_type&>)
  _CCCL_HOST_API __base_mdarray(extents_type __ext, allocator_type __allocator)
      : __base_mdarray{mapping_type{__ext}, __allocator}
  {}

  _CCCL_TEMPLATE(typename... _OtherIndexTypes)
  _CCCL_REQUIRES(__constraints::template __can_construct_from_handle_and_variadic<_OtherIndexTypes...>)
  _CCCL_HOST_API __base_mdarray(_OtherIndexTypes... __exts)
      : __base_mdarray{extents_type{static_cast<::cuda::std::size_t>(__exts)...}}
  {}

  _CCCL_TEMPLATE(typename _OtherIndexType, ::cuda::std::size_t _Size)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__matches_dynamic_rank<extents_type, _Size> _CCCL_AND
                   __constraints::template __is_constructible_from_index_type<_OtherIndexType>)
  _CCCL_HOST_API __base_mdarray(const ::cuda::std::array<_OtherIndexType, _Size>& __exts)
      : __base_mdarray{extents_type{__exts}, _Derived::__get_default_allocator()}
  {}

  _CCCL_TEMPLATE(typename _OtherIndexType, ::cuda::std::size_t _Size)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__matches_static_rank<extents_type, _Size> _CCCL_AND
                   __constraints::template __is_constructible_from_index_type<_OtherIndexType>)
  _CCCL_HOST_API __base_mdarray(const ::cuda::std::array<_OtherIndexType, _Size>& __exts)
      : __base_mdarray{extents_type{__exts}, _Derived::__get_default_allocator()}
  {}

  _CCCL_TEMPLATE(typename _OtherIndexType, ::cuda::std::size_t _Size)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__matches_dynamic_rank<extents_type, _Size> _CCCL_AND
                   __constraints::template __is_constructible_from_index_type<_OtherIndexType>)
  _CCCL_HOST_API __base_mdarray(const ::cuda::std::array<_OtherIndexType, _Size>& __exts, allocator_type __allocator)
      : __base_mdarray{extents_type{__exts}, __allocator}
  {}

  _CCCL_TEMPLATE(typename _OtherIndexType, ::cuda::std::size_t _Size)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__matches_static_rank<extents_type, _Size> _CCCL_AND
                   __constraints::template __is_constructible_from_index_type<_OtherIndexType>)
  _CCCL_HOST_API __base_mdarray(const ::cuda::std::array<_OtherIndexType, _Size>& __exts, allocator_type __allocator)
      : __base_mdarray{extents_type{__exts}, __allocator}
  {}

  _CCCL_TEMPLATE(typename _OtherIndexType, ::cuda::std::size_t _Size)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__matches_dynamic_rank<extents_type, _Size> _CCCL_AND
                   __constraints::template __is_constructible_from_index_type<_OtherIndexType>)
  _CCCL_HOST_API __base_mdarray(::cuda::std::span<_OtherIndexType, _Size> __exts)
      : __base_mdarray{extents_type{__exts}, _Derived::__get_default_allocator()}
  {}

  _CCCL_TEMPLATE(typename _OtherIndexType, ::cuda::std::size_t _Size)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__matches_static_rank<extents_type, _Size> _CCCL_AND
                   __constraints::template __is_constructible_from_index_type<_OtherIndexType>)
  _CCCL_HOST_API __base_mdarray(::cuda::std::span<_OtherIndexType, _Size> __exts)
      : __base_mdarray{extents_type{__exts}, _Derived::__get_default_allocator()}
  {}

  _CCCL_TEMPLATE(typename _OtherIndexType, ::cuda::std::size_t _Size)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__matches_dynamic_rank<extents_type, _Size> _CCCL_AND
                   __constraints::template __is_constructible_from_index_type<_OtherIndexType>)
  _CCCL_HOST_API __base_mdarray(::cuda::std::span<_OtherIndexType, _Size> __exts, allocator_type __allocator)
      : __base_mdarray{extents_type{__exts}, __allocator}
  {}

  _CCCL_TEMPLATE(typename _OtherIndexType, ::cuda::std::size_t _Size)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__matches_static_rank<extents_type, _Size> _CCCL_AND
                   __constraints::template __is_constructible_from_index_type<_OtherIndexType>)
  _CCCL_HOST_API __base_mdarray(::cuda::std::span<_OtherIndexType, _Size> __exts, allocator_type __allocator)
      : __base_mdarray{extents_type{__exts}, __allocator}
  {}

  _CCCL_HOST_API __base_mdarray(const __base_mdarray& __other)
      : __base_mdarray{__other, __other.__get_allocator()}
  {}

  _CCCL_HOST_API __base_mdarray(const __base_mdarray& __other, ::cuda::std::type_identity_t<allocator_type> __allocator)
      : __allocator_base{__allocator}
      , mdspan_type{nullptr, __other.mapping()}
  {
    if (__other.data_handle() != nullptr)
    {
      const auto __size                = ::cuda::experimental::__mapping_size_bytes(__other.view());
      auto __new_data                  = static_cast<pointer>(this->__get_allocator().allocate_sync(__size));
      static_cast<mdspan_type&>(*this) = mdspan_type{__new_data, __other.mapping()};
      static_cast<_Derived&>(*this).__copy_from(__other.view());
    }
  }

  _CCCL_HOST_API __base_mdarray(__base_mdarray&& __other) noexcept
      : __allocator_base{::cuda::std::move(static_cast<__allocator_base&>(__other))}
      , mdspan_type{::cuda::std::exchange(static_cast<mdspan_type&>(__other), mdspan_type{})}
  {}

  _CCCL_HOST_API __base_mdarray& operator=(const __base_mdarray& __other)
  {
    if (this == &__other)
    {
      return *this;
    }
    return __assign_from(__other.view(), __other.__get_allocator());
  }

  template <typename _OtherElementType, typename _OtherExtents, typename _OtherLayoutPolicy>
  _CCCL_HOST_API __base_mdarray&
  operator=(const _ViewType<_OtherElementType, _OtherExtents, _OtherLayoutPolicy>& __other_view)
  {
    return __assign_from(__other_view.view(), this->__get_allocator());
  }

  template <typename _OtherElementType, typename _OtherExtents, typename _OtherLayoutPolicy>
  _CCCL_HOST_API __base_mdarray& __assign_from(
    const _ViewType<_OtherElementType, _OtherExtents, _OtherLayoutPolicy>& __other_view, _Allocator __other_allocator)
  {
    const auto __size_other   = ::cuda::experimental::__mapping_size_bytes(__other_view);
    const auto __size_current = ::cuda::experimental::__mapping_size_bytes(this->view());
    const bool __realloc      = __size_current != __size_other || this->__get_allocator() != __other_allocator;
    if (__realloc)
    {
      __release_storage();
      this->__get_allocator() = __other_allocator;
      if (__other_view.data_handle() != nullptr)
      {
        auto __new_data                  = static_cast<pointer>(this->__get_allocator().allocate_sync(__size_other));
        static_cast<mdspan_type&>(*this) = mdspan_type{__new_data, __other_view.mapping()};
      }
      else
      {
        static_cast<mdspan_type&>(*this) = mdspan_type{nullptr, __other_view.mapping()};
      }
    }
    else // no reallocation needed
    {
      pointer __new_data = this->data_handle();
      if (__other_view.data_handle() == nullptr)
      {
        __release_storage();
        __new_data = nullptr;
      }
      else if (this->data_handle() == nullptr)
      {
        __new_data = static_cast<pointer>(this->__get_allocator().allocate_sync(__size_other));
      }
      static_cast<mdspan_type&>(*this) = mdspan_type{__new_data, __other_view.mapping()};
    }
    if (__other_view.data_handle() != nullptr)
    {
      static_cast<_Derived&>(*this).__copy_from(__other_view);
    }
    return *this;
  }

  _CCCL_HOST_API
  __base_mdarray(__base_mdarray&& __other, const ::cuda::std::type_identity_t<allocator_type> __allocator) noexcept
      : __allocator_base{__allocator}
      , mdspan_type{::cuda::std::exchange(static_cast<mdspan_type&>(__other), mdspan_type{})}
  {}

  _CCCL_HIDE_FROM_ABI __base_mdarray& operator=(__base_mdarray&& __other) noexcept
  {
    if (this == &__other)
    {
      return *this;
    }
    __release_storage();
    static_cast<mdspan_type&>(*this) = ::cuda::std::exchange(static_cast<mdspan_type&>(__other), mdspan_type{});
    this->__get_allocator()          = ::cuda::std::move(__other.__get_allocator());
    return *this;
  }

  _CCCL_HOST_API ~__base_mdarray() noexcept
  {
    __release_storage();
  }

  [[nodiscard]] _CCCL_HOST_API view_type view() noexcept
  {
    return static_cast<view_type&>(*this);
  }

  [[nodiscard]] _CCCL_HOST_API const_view_type view() const noexcept
  {
    return static_cast<const_view_type>(static_cast<const view_type&>(*this));
  }

  [[nodiscard]] _CCCL_HOST_API operator view_type() noexcept
  {
    return static_cast<view_type&>(*this);
  }

  [[nodiscard]] _CCCL_HOST_API operator const_view_type() const noexcept
  {
    return static_cast<const_view_type>(static_cast<const view_type&>(*this));
  }
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDAX__CONTAINER_MDARRAY_BASE_CUH
