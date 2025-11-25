//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINER_MDARRAY__
#define __CUDAX__CONTAINER_MDARRAY__

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__mdspan/mdspan.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/memory>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
namespace __mdarray_detail
{
template <class _Extents>
struct __extents_traits
{
  using size_type = typename _Extents::size_type;
  using rank_type = typename _Extents::rank_type;

  static constexpr ::cuda::std::size_t __rank_v = static_cast<::cuda::std::size_t>(_Extents::rank());

private:
  template <::cuda::std::size_t... _Idx>
  [[nodiscard]]
  _CCCL_API static constexpr bool __has_zero_static_extent_impl(::cuda::std::index_sequence<_Idx...>) noexcept
  {
    if constexpr (sizeof...(_Idx) == 0)
    {
      return false;
    }
    else
    {
      return ((_Extents::static_extent(static_cast<rank_type>(_Idx)) == 0) || ...);
    }
  }

public:
  static constexpr bool __has_dynamic_extents = (_Extents::rank_dynamic() != 0);
  static constexpr bool __has_zero_static_extent =
    __has_zero_static_extent_impl(::cuda::std::make_index_sequence<__rank_v>{});
  static constexpr bool __allow_default_construct =
    (_Extents::rank() == 0) || __has_dynamic_extents || (__rank_v != 0 && __has_zero_static_extent);
  static constexpr bool __default_extents_zero = __has_dynamic_extents || (__rank_v != 0 && __has_zero_static_extent);

  template <class _E>
  [[nodiscard]] _CCCL_API static constexpr bool __is_zero_sized(const _E& __ext) noexcept
  {
    if constexpr (_Extents::rank() == 0)
    {
      return false;
    }

    for (rank_type __r = 0; __r < _Extents::rank(); ++__r)
    {
      if (__ext.extent(__r) == typename _Extents::index_type{0})
      {
        return true;
      }
    }
    return false;
  }
};
} // namespace __mdarray_detail

template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy   = ::cuda::std::layout_right,
          typename _AccessorPolicy = ::cuda::std::default_accessor<_ElementType>,
          typename _Allocator      = ::cuda::std::allocator<_ElementType[]>>
class mdarray
{
public:
  using extents_type     = _Extents;
  using layout_type      = _LayoutPolicy;
  using accessor_type    = _AccessorPolicy;
  using mapping_type     = typename layout_type::template mapping<extents_type>;
  using element_type     = _ElementType;
  using value_type       = ::cuda::std::remove_cv_t<element_type>;
  using index_type       = typename extents_type::index_type;
  using size_type        = typename extents_type::size_type;
  using rank_type        = typename extents_type::rank_type;
  using data_handle_type = typename accessor_type::data_handle_type;
  using reference        = typename accessor_type::reference;
  using pointer          = element_type*;

private:
  using __traits                                = __mdarray_detail::__extents_traits<extents_type>;
  static constexpr bool __can_default_construct = __traits::__allow_default_construct;
  static constexpr bool __mapping_constructible = ::cuda::std::is_constructible_v<mapping_type, const extents_type&>;
  static constexpr bool __allow_zero_mapping    = __mapping_constructible && __traits::__default_extents_zero;
  static constexpr ::cuda::std::size_t __rank_v = static_cast<::cuda::std::size_t>(extents_type::rank());

public:
  _CCCL_API mdarray() _CCCL_REQUIRES(__can_default_construct) = default;

  mdarray(const mdarray&)            = delete;
  mdarray& operator=(const mdarray&) = delete;
  ~mdarray()                         = default;

  _CCCL_API mdarray(mdarray&& __other) noexcept _CCCL_REQUIRES(__can_default_construct&& __mapping_constructible)
      : __ptr_(::cuda::std::move(__other.__ptr_))
      , __mapping_(::cuda::std::move(__other.__mapping_))
  {
    __other.__mapping_ = mapping_type{extents_type{}};
  }

  _CCCL_API mdarray& operator=(mdarray&& __other) noexcept
    _CCCL_REQUIRES(__can_default_construct&& __mapping_constructible)
  {
    if (this == &__other)
    {
      return *this;
    }
    __ptr_             = ::cuda::std::move(__other.__ptr_);
    __mapping_         = ::cuda::std::move(__other.__mapping_);
    __other.__mapping_ = mapping_type{extents_type{}};
    return *this;
  }

  template <::cuda::std::size_t _InputExtent>
  _CCCL_API
  mdarray(::cuda::std::span<element_type, _InputExtent> __sp, const mapping_type& __mapping, deleter_type __deleter)
      : __ptr_(__sp.data(), ::cuda::std::move(__deleter))
      , __mapping_(__mapping)
  {
    _CCCL_ASSERT(static_cast<::cuda::std::size_t>(__mapping_.required_span_size()) <= __sp.size(),
                 "mdarray: span too small for requested mapping.");
  }

  template <::cuda::std::size_t _InputExtent>
  _CCCL_API
  mdarray(::cuda::std::span<element_type, _InputExtent> __sp, const extents_type& __ext, deleter_type __deleter)
    requires(__mapping_constructible)
      : mdarray(__sp, mapping_type{__ext}, ::cuda::std::move(__deleter))
  {}

  template <::cuda::std::size_t _InputExtent>
  _CCCL_API mdarray(::cuda::std::span<element_type, _InputExtent> __sp, const mapping_type& __mapping)
    requires(::cuda::std::is_nothrow_default_constructible_v<deleter_type>)
      : mdarray(__sp, __mapping, deleter_type{})
  {}

  template <::cuda::std::size_t _InputExtent>
  _CCCL_API mdarray(::cuda::std::span<element_type, _InputExtent> __sp, const extents_type& __ext)
    requires(__mapping_constructible && ::cuda::std::is_nothrow_default_constructible_v<deleter_type>)
      : mdarray(__sp, __ext, deleter_type{})
  {}

  template <::cuda::std::size_t _InputExtent, class... _OtherIndexTypes>
  _CCCL_API mdarray(::cuda::std::span<element_type, _InputExtent> __sp, _OtherIndexTypes... __exts)
    requires(::cuda::std::is_nothrow_default_constructible_v<deleter_type>
             && (::cuda::std::is_convertible_v<_OtherIndexTypes, index_type> && ...)
             && (::cuda::std::is_nothrow_constructible_v<index_type, _OtherIndexTypes> && ...)
             && (sizeof...(_OtherIndexTypes) == __rank_v
                 || sizeof...(_OtherIndexTypes) == static_cast<::cuda::std::size_t>(extents_type::rank_dynamic()))
             && __mapping_constructible)
      : mdarray(__sp, extents_type{static_cast<index_type>(::cuda::std::move(__exts))...}, deleter_type{})
  {}

  template <::cuda::std::size_t _InputExtent>
  _CCCL_API explicit mdarray(::cuda::std::span<element_type, _InputExtent> __sp)
    requires(__mapping_constructible && ::cuda::std::is_nothrow_default_constructible_v<deleter_type>)
      : mdarray(__sp, extents_type{}, deleter_type{})
  {}

  _CCCL_API mdarray(::cuda::std::nullptr_t) noexcept
    requires(::cuda::std::is_nothrow_default_constructible_v<deleter_type> && __allow_zero_mapping
             && ::cuda::std::is_constructible_v<__unique_ptr, ::cuda::std::nullptr_t>)
      : __ptr_(nullptr)
      , __mapping_{extents_type{}}
  {}

  [[nodiscard]] _CCCL_API pointer get() const noexcept
  {
    return __ptr_.get();
  }

  [[nodiscard]] _CCCL_API explicit operator bool() const noexcept
  {
    return static_cast<bool>(__ptr_);
  }

  friend _CCCL_API void swap(mdarray& __lhs, mdarray& __rhs) noexcept
    requires(::cuda::std::is_swappable_v<deleter_type>)
  {
    ::cuda::std::swap(__lhs.__ptr_, __rhs.__ptr_);
    ::cuda::std::swap(__lhs.__mapping_, __rhs.__mapping_);
  }

  _CCCL_API
  operator ::cuda::std::mdspan<element_type, extents_type, layout_type, ::cuda::std::default_accessor<element_type>>()
    const
  {
    return {__ptr_.get(), __mapping_};
  }

  [[nodiscard]] _CCCL_API pointer release() noexcept
    requires(__allow_zero_mapping)
  {
    pointer __released = __ptr_.release();
    __mapping_         = mapping_type{extents_type{}};
    return __released;
  }

  _CCCL_API void reset() noexcept
    requires(__allow_zero_mapping)
  {
    __ptr_.reset();
    __mapping_ = mapping_type{extents_type{}};
  }

  _CCCL_API void reset(::cuda::std::type_identity_t<pointer> __ptr) noexcept
  {
    __ptr_.reset(__ptr);
  }

  //--------------------------------------------------------------------------------------------------------------------
  // extents() and rank()

  [[nodiscard]] _CCCL_API static constexpr rank_type rank() noexcept
  {
    return extents_type::rank();
  }

  [[nodiscard]] _CCCL_API static constexpr rank_type rank_dynamic() noexcept
  {
    return extents_type::rank_dynamic();
  }

  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t static_extent(rank_type __r) noexcept
  {
    return extents_type::static_extent(__r);
  }

  [[nodiscard]] _CCCL_API index_type extent(rank_type __r) const noexcept
  {
    return mapping().extents().extent(__r);
  }

  //--------------------------------------------------------------------------------------------------------------------
  // access operators

#if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
  _CCCL_TEMPLATE(class... _OtherIndexTypes)
  _CCCL_REQUIRES((sizeof...(_OtherIndexTypes) == extents_type::rank())
                   _CCCL_AND __mdspan_detail::__all_convertible_to_index_type<index_type, _OtherIndexTypes...>)
  [[nodiscard]] _CCCL_API constexpr reference operator[](_OtherIndexTypes... __indices) const
  {
    // Note the standard layouts would also check this, but user provided ones may not, so we
    // check the precondition here
    _CCCL_ASSERT(__mdspan_detail::__is_multidimensional_index_in(extents(), __indices...),
                 "mdspan: operator[] out of bounds access");
    return accessor().access(data_handle(), mapping()(static_cast<index_type>(::cuda::std::move(__indices))...));
  }
#else
  _CCCL_TEMPLATE(class _OtherIndexType)
  _CCCL_REQUIRES((extents_type::rank() == 1) _CCCL_AND is_convertible_v<_OtherIndexType, index_type> _CCCL_AND
                   is_nothrow_constructible_v<index_type, _OtherIndexType>)
  [[nodiscard]] _CCCL_API constexpr reference operator[](_OtherIndexType __index) const
  {
    return accessor().access(data_handle(), mapping()(static_cast<index_type>(::cuda::std::move(__index))));
  }
#endif // _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS

  template <class _OtherIndexType, size_t... _Idxs>
  [[nodiscard]] _CCCL_API constexpr decltype(auto)
  __op_bracket(const array<_OtherIndexType, _Extents::rank()>& __indices, index_sequence<_Idxs...>) const noexcept
  {
    // Note the standard layouts would also check this, but user provided ones may not, so we
    // check the precondition here
    _CCCL_ASSERT(__mdspan_detail::__is_multidimensional_index_in(extents(), __indices[_Idxs]...),
                 "mdspan: operator[array] out of bounds access");
    return mapping()(__indices[_Idxs]...);
  }

  template <class _OtherIndexType, size_t... _Idxs>
  [[nodiscard]] _CCCL_API constexpr decltype(auto)
  __op_bracket(span<_OtherIndexType, _Extents::rank()> __indices, index_sequence<_Idxs...>) const noexcept
  {
    // Note the standard layouts would also check this, but user provided ones may not, so we
    // check the precondition here
    _CCCL_ASSERT(__mdspan_detail::__is_multidimensional_index_in(extents(), __indices[_Idxs]...),
                 "mdspan: operator[span] out of bounds access");
    return mapping()(__indices[_Idxs]...);
  }

  _CCCL_TEMPLATE(class _OtherIndexType)
  _CCCL_REQUIRES(is_convertible_v<const _OtherIndexType&, index_type> _CCCL_AND
                   is_nothrow_constructible_v<index_type, const _OtherIndexType&>)
  [[nodiscard]] _CCCL_API constexpr reference
  operator[](const array<_OtherIndexType, extents_type::rank()>& __indices) const
  {
    return accessor().access(data_handle(), __op_bracket(__indices, make_index_sequence<rank()>()));
  }

  _CCCL_TEMPLATE(class _OtherIndexType)
  _CCCL_REQUIRES(is_convertible_v<const _OtherIndexType&, index_type> _CCCL_AND
                   is_nothrow_constructible_v<index_type, const _OtherIndexType&>)
  [[nodiscard]] _CCCL_API constexpr reference operator[](span<_OtherIndexType, extents_type::rank()> __indices) const
  {
    return accessor().access(data_handle(), __op_bracket(__indices, make_index_sequence<rank()>()));
  }

  //! Nonstandard extension to no break our users too hard
  _CCCL_TEMPLATE(class... _Indices)
  _CCCL_REQUIRES(__mdspan_detail::__all_convertible_to_index_type<index_type, _Indices...>)
  [[nodiscard]] _CCCL_API constexpr reference operator()(_Indices... __indices) const
  {
    // Note the standard layouts would also check this, but user provided ones may not, so we
    // check the precondition here
    _CCCL_ASSERT(__mdspan_detail::__is_multidimensional_index_in(extents(), __indices...),
                 "mdspan: operator() out of bounds access");
    return accessor().access(data_handle(), mapping()(__indices...));
  }

  //--------------------------------------------------------------------------------------------------------------------
  // size()

  [[nodiscard]] _CCCL_API static constexpr bool
  __mul_overflow(::cuda::std::size_t x, ::cuda::std::size_t y, ::cuda::std::size_t* res) noexcept
  {
    *res = x * y;
    return x && ((*res / x) != y);
  }

  template <::cuda::std::size_t... _Idxs>
  [[nodiscard]] _CCCL_API constexpr bool __check_size() const noexcept
  {
    ::cuda::std::size_t __prod = 1;
    for (size_t __r = 0; __r != extents_type::rank(); ++__r)
    {
      if (__mul_overflow(__prod, mapping().extents().extent(__r), &__prod))
      {
        return false;
      }
    }
    return true;
  }

  template <::cuda::std::size_t... _Idxs>
  [[nodiscard]] _CCCL_API constexpr size_type __op_size(::cuda::std::index_sequence<_Idxs...>) const noexcept
  {
    return (size_type{1} * ... * static_cast<size_type>(mapping().extents().extent(_Idxs)));
  }

  [[nodiscard]] _CCCL_API constexpr size_type size() const noexcept
  {
    // Could leave this as only checked in debug mode: semantically size() is never
    // guaranteed to be related to any accessible range
    _CCCL_ASSERT(__check_size(), "mdspan: size() is not representable as size_type");
    return __op_size(make_index_sequence<rank()>());
  }

  //--------------------------------------------------------------------------------------------------------------------
  // other methods

  template <::cuda::std::size_t... _Idxs>
  [[nodiscard]] _CCCL_API constexpr bool __op_empty(::cuda::std::index_sequence<_Idxs...>) const noexcept
  {
    return (((mapping().extents().extent(_Idxs) == index_type{0})) || ...);
  }

  [[nodiscard]] _CCCL_API constexpr bool empty() const noexcept
  {
    return __op_empty(::cuda::std::make_index_sequence<rank()>());
  }

  [[nodiscard]] _CCCL_API constexpr const extents_type& extents() const noexcept
  {
    return mapping().extents();
  }
  [[nodiscard]] _CCCL_API constexpr const data_handle_type& data_handle() const noexcept
  {
    return this->template __get<0>();
  }
  [[nodiscard]] _CCCL_API constexpr const mapping_type& mapping() const noexcept
  {
    return this->template __get<1>();
  }
  [[nodiscard]] _CCCL_API constexpr const accessor_type& accessor() const noexcept
  {
    return this->template __get<2>();
  }

  [[nodiscard]] _CCCL_API static constexpr bool is_always_unique() noexcept(noexcept(mapping_type::is_always_unique()))
  {
    return mapping_type::is_always_unique();
  }
  [[nodiscard]] _CCCL_API static constexpr bool
  is_always_exhaustive() noexcept(noexcept(mapping_type::is_always_exhaustive()))
  {
    return mapping_type::is_always_exhaustive();
  }
  [[nodiscard]] _CCCL_API static constexpr bool is_always_strided() noexcept(noexcept(mapping_type::is_always_strided()))
  {
    return mapping_type::is_always_strided();
  }

  [[nodiscard]] _CCCL_API constexpr bool is_unique() const
    noexcept(noexcept(::cuda::std::declval<const mapping_type&>().is_unique()))
  {
    return mapping().is_unique();
  }
  [[nodiscard]] _CCCL_API constexpr bool is_exhaustive() const
    noexcept(noexcept(::cuda::std::declval<const mapping_type&>().is_exhaustive()))
  {
    auto __tmp = mapping(); // workaround for clang with nodiscard
    return __tmp.is_exhaustive();
  }
  [[nodiscard]] _CCCL_API constexpr bool is_strided() const
    noexcept(noexcept(::cuda::std::declval<const mapping_type&>().is_strided()))
  {
    return mapping().is_strided();
  }
  [[nodiscard]] _CCCL_API constexpr index_type stride(rank_type __r) const
  {
    return mapping().stride(__r);
  }
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDAX__CONTAINER_MDARRAY__
