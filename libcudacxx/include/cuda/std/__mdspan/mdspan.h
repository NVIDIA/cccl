// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_MDSPAN_HPP
#define _LIBCUDACXX___MDSPAN_MDSPAN_HPP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/mdspan.h>
#include <cuda/std/__mdspan/concepts.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__mdspan/layout_right.h>
#include <cuda/std/__type_traits/extent.h>
#include <cuda/std/__type_traits/is_abstract.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/rank.h>
#include <cuda/std/__type_traits/remove_all_extents.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/as_const.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/span>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _ElementType,
          class _Extents,
          class _LayoutPolicy   = layout_right,
          class _AccessorPolicy = default_accessor<_ElementType>>
class mdspan
{
private:
  static_assert(__mdspan_detail::__is_extents_v<_Extents>,
                "mdspan: Extents template parameter must be a specialization of extents.");
  static_assert(!_CCCL_TRAIT(is_array, _ElementType),
                "mdspan: ElementType template parameter may not be an array type");
  static_assert(!_CCCL_TRAIT(is_abstract, _ElementType),
                "mdspan: ElementType template parameter may not be an abstract class");
  static_assert(_CCCL_TRAIT(is_same, _ElementType, typename _AccessorPolicy::element_type),
                "mdspan: ElementType template parameter must match AccessorPolicy::element_type");
  static_assert(__mdspan_detail::__is_valid_layout_mapping<_LayoutPolicy, _Extents>,
                "mdspan: LayoutPolicy template parameter is invalid. A common mistake is to pass a layout mapping "
                "instead of a layout policy");

public:
  using extents_type     = _Extents;
  using layout_type      = _LayoutPolicy;
  using accessor_type    = _AccessorPolicy;
  using mapping_type     = typename layout_type::template mapping<extents_type>;
  using element_type     = _ElementType;
  using value_type       = remove_cv_t<element_type>;
  using index_type       = typename extents_type::index_type;
  using size_type        = typename extents_type::size_type;
  using rank_type        = typename extents_type::rank_type;
  using data_handle_type = typename accessor_type::data_handle_type;
  using reference        = typename accessor_type::reference;

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr rank_type rank() noexcept
  {
    return extents_type::rank();
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr rank_type rank_dynamic() noexcept
  {
    return extents_type::rank_dynamic();
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t static_extent(rank_type __r) noexcept
  {
    return extents_type::static_extent(__r);
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type extent(rank_type __r) const noexcept
  {
    return __map_.extents().extent(__r);
  };

public:
  //--------------------------------------------------------------------------------
  // [mdspan.mdspan.cons], mdspan constructors, assignment, and destructor

#if _CCCL_STD_VER >= 2020
  constexpr mdspan()
    requires((extents_type::rank_dynamic() > 0) && is_default_constructible_v<data_handle_type>
             && is_default_constructible_v<mapping_type> && is_default_constructible_v<accessor_type>)
  = default;
#else // ^^^ _CCCL_STD_VER >= 2020 ^^^ / vvv _CCCL_STD_VER <= 2017 vvv
  _CCCL_TEMPLATE(class _Extents2 = _Extents)
  _CCCL_REQUIRES((_Extents2::rank_dynamic() > 0) //
                 _CCCL_AND _CCCL_TRAIT(is_default_constructible, data_handle_type)
                   _CCCL_AND _CCCL_TRAIT(is_default_constructible, mapping_type)
                     _CCCL_AND _CCCL_TRAIT(is_default_constructible, accessor_type))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan() noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, data_handle_type)
    && _CCCL_TRAIT(is_nothrow_default_constructible, mapping_type)
    && _CCCL_TRAIT(is_nothrow_default_constructible, accessor_type))
  {}
#endif // _CCCL_STD_VER <= 2017

  constexpr mdspan(const mdspan&) = default;
  constexpr mdspan(mdspan&&)      = default;

  _CCCL_TEMPLATE(class... _OtherIndexTypes)
  _CCCL_REQUIRES(((sizeof...(_OtherIndexTypes) == rank()) || (sizeof...(_OtherIndexTypes) == rank_dynamic()))
                   _CCCL_AND _CCCL_FOLD_AND(_CCCL_TRAIT(is_convertible, _OtherIndexTypes, index_type))
                     _CCCL_AND _CCCL_FOLD_AND(_CCCL_TRAIT(is_nothrow_constructible, index_type, _OtherIndexTypes))
                       _CCCL_AND _CCCL_TRAIT(is_constructible, mapping_type, extents_type)
                         _CCCL_AND _CCCL_TRAIT(is_default_constructible, accessor_type))
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr mdspan(data_handle_type __p, _OtherIndexTypes... __exts)
      : __ptr_(_CUDA_VSTD::move(__p))
      , __map_(extents_type(static_cast<index_type>(_CUDA_VSTD::move(__exts))...))
      , __acc_{}
  {}

  template <class _OtherIndexType>
  static constexpr bool __is_constructible_from =
    _CCCL_TRAIT(is_convertible, const _OtherIndexType&, index_type)
    && _CCCL_TRAIT(is_nothrow_constructible, index_type, const _OtherIndexType&)
    && _CCCL_TRAIT(is_constructible, mapping_type, extents_type)
    && _CCCL_TRAIT(is_default_constructible, accessor_type);

  _CCCL_TEMPLATE(class _OtherIndexType, size_t _Size)
  _CCCL_REQUIRES((_Size == rank_dynamic()) _CCCL_AND __is_constructible_from<_OtherIndexType>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(data_handle_type __p, const array<_OtherIndexType, _Size>& __exts)
      : __ptr_(_CUDA_VSTD::move(__p))
      , __map_(extents_type{__exts})
      , __acc_{}
  {}

  _CCCL_TEMPLATE(class _OtherIndexType, size_t _Size)
  _CCCL_REQUIRES((_Size == rank()) _CCCL_AND(_Size != rank_dynamic())
                   _CCCL_AND __is_constructible_from<_OtherIndexType>)
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr mdspan(data_handle_type __p, const array<_OtherIndexType, _Size>& __exts)
      : __ptr_(_CUDA_VSTD::move(__p))
      , __map_(extents_type{__exts})
      , __acc_{}
  {}

  _CCCL_TEMPLATE(class _OtherIndexType, size_t _Size)
  _CCCL_REQUIRES((_Size == rank_dynamic()) _CCCL_AND __is_constructible_from<_OtherIndexType>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(data_handle_type __p, span<_OtherIndexType, _Size> __exts)
      : __ptr_(_CUDA_VSTD::move(__p))
      , __map_(extents_type{__exts})
      , __acc_{}
  {}

  _CCCL_TEMPLATE(class _OtherIndexType, size_t _Size)
  _CCCL_REQUIRES((_Size == rank()) _CCCL_AND(_Size != rank_dynamic())
                   _CCCL_AND __is_constructible_from<_OtherIndexType>)
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr mdspan(data_handle_type __p, span<_OtherIndexType, _Size> __exts)
      : __ptr_(_CUDA_VSTD::move(__p))
      , __map_(extents_type{__exts})
      , __acc_{}
  {}

  _CCCL_TEMPLATE(class _AccessorPolicy2 = _AccessorPolicy)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_default_constructible, _AccessorPolicy2)
                   _CCCL_AND _CCCL_TRAIT(is_constructible, mapping_type, const extents_type&))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(data_handle_type __p, const extents_type& __exts)
      : __ptr_(_CUDA_VSTD::move(__p))
      , __map_(__exts)
      , __acc_{}
  {}

  _CCCL_TEMPLATE(class _AccessorPolicy2 = _AccessorPolicy)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_default_constructible, _AccessorPolicy2))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(data_handle_type __p, const mapping_type& __m)
      : __ptr_(_CUDA_VSTD::move(__p))
      , __map_(__m)
      , __acc_{}
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(data_handle_type __p, const mapping_type& __m, const accessor_type& __a)
      : __ptr_(_CUDA_VSTD::move(__p))
      , __map_(__m)
      , __acc_(__a)
  {}

  template <class _OtherExtents, class _OtherLayoutPolicy, class _OtherAccessor>
  static constexpr bool __is_convertible_from =
    _CCCL_TRAIT(is_constructible, mapping_type, const typename _OtherLayoutPolicy::template mapping<_OtherExtents>&)
    && _CCCL_TRAIT(is_constructible, accessor_type, const _OtherAccessor&);

  template <class _OtherExtents, class _OtherLayoutPolicy, class _OtherAccessor>
  static constexpr bool __is_implicit_convertible_from =
    _CCCL_TRAIT(is_convertible, const typename _OtherLayoutPolicy::template mapping<_OtherExtents>&, mapping_type)
    && _CCCL_TRAIT(is_convertible, const _OtherAccessor&, accessor_type);

  _CCCL_TEMPLATE(class _OtherElementType, class _OtherExtents, class _OtherLayoutPolicy, class _OtherAccessor)
  _CCCL_REQUIRES((rank() > 0) //
                 _CCCL_AND __is_convertible_from<_OtherExtents, _OtherLayoutPolicy, _OtherAccessor> //
                   _CCCL_AND __is_implicit_convertible_from<_OtherExtents, _OtherLayoutPolicy, _OtherAccessor>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(
    const mdspan<_OtherElementType, _OtherExtents, _OtherLayoutPolicy, _OtherAccessor>& __other)
      : __ptr_(__other.__ptr_)
      , __map_(__other.__map_)
      , __acc_(__other.__acc_)
  {
    static_assert(_CCCL_TRAIT(is_constructible, data_handle_type, const typename _OtherAccessor::data_handle_type&),
                  "mdspan: incompatible data_handle_type for mdspan construction");
    static_assert(_CCCL_TRAIT(is_constructible, extents_type, _OtherExtents),
                  "mdspan: incompatible extents for mdspan construction");

    // The following precondition is part of the standard, but is unlikely to be triggered.
    // The extents constructor checks this and the mapping must be storing the extents, since
    // its extents() function returns a const reference to extents_type.
    // The only way this can be triggered is if the mapping conversion constructor would for example
    // always construct its extents() only from the dynamic extents, instead of from the other extents.
    for (size_t __r = 0; __r < rank(); __r++)
    {
      // Not catching this could lead to out of bounds errors later
      // e.g. mdspan<int, dextents<char,1>, non_checking_layout> m =
      //        mdspan<int, dextents<unsigned, 1>, non_checking_layout>(ptr, 200); leads to an extent of -56 on m
      _CCCL_ASSERT((static_extent(__r) == dynamic_extent)
                     || (static_cast<index_type>(__other.extent(__r)) == static_cast<index_type>(static_extent(__r))),
                   "mdspan: conversion mismatch of source dynamic extents with static extents");
    }
  }

  _CCCL_TEMPLATE(class _OtherElementType, class _OtherExtents, class _OtherLayoutPolicy, class _OtherAccessor)
  _CCCL_REQUIRES((rank() == 0) //
                 _CCCL_AND __is_convertible_from<_OtherExtents, _OtherLayoutPolicy, _OtherAccessor> //
                   _CCCL_AND __is_implicit_convertible_from<_OtherExtents, _OtherLayoutPolicy, _OtherAccessor>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(
    const mdspan<_OtherElementType, _OtherExtents, _OtherLayoutPolicy, _OtherAccessor>& __other)
      : __ptr_(__other.__ptr_)
      , __map_(__other.__map_)
      , __acc_(__other.__acc_)
  {
    static_assert(_CCCL_TRAIT(is_constructible, data_handle_type, const typename _OtherAccessor::data_handle_type&),
                  "mdspan: incompatible data_handle_type for mdspan construction");
    static_assert(_CCCL_TRAIT(is_constructible, extents_type, _OtherExtents),
                  "mdspan: incompatible extents for mdspan construction");
  }
  _CCCL_TEMPLATE(class _OtherElementType, class _OtherExtents, class _OtherLayoutPolicy, class _OtherAccessor)
  _CCCL_REQUIRES((rank() > 0) //
                 _CCCL_AND __is_convertible_from<_OtherExtents, _OtherLayoutPolicy, _OtherAccessor> //
                   _CCCL_AND(!__is_implicit_convertible_from<_OtherExtents, _OtherLayoutPolicy, _OtherAccessor>))
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr mdspan(
    const mdspan<_OtherElementType, _OtherExtents, _OtherLayoutPolicy, _OtherAccessor>& __other)
      : __ptr_(__other.__ptr_)
      , __map_(__other.__map_)
      , __acc_(__other.__acc_)
  {
    static_assert(_CCCL_TRAIT(is_constructible, data_handle_type, const typename _OtherAccessor::data_handle_type&),
                  "mdspan: incompatible data_handle_type for mdspan construction");
    static_assert(_CCCL_TRAIT(is_constructible, extents_type, _OtherExtents),
                  "mdspan: incompatible extents for mdspan construction");

    // The following precondition is part of the standard, but is unlikely to be triggered.
    // The extents constructor checks this and the mapping must be storing the extents, since
    // its extents() function returns a const reference to extents_type.
    // The only way this can be triggered is if the mapping conversion constructor would for example
    // always construct its extents() only from the dynamic extents, instead of from the other extents.
    for (size_t __r = 0; __r < rank(); __r++)
    {
      // Not catching this could lead to out of bounds errors later
      // e.g. mdspan<int, dextents<char,1>, non_checking_layout> m =
      //        mdspan<int, dextents<unsigned, 1>, non_checking_layout>(ptr, 200); leads to an extent of -56 on m
      _CCCL_ASSERT((static_extent(__r) == dynamic_extent)
                     || (static_cast<index_type>(__other.extent(__r)) == static_cast<index_type>(static_extent(__r))),
                   "mdspan: conversion mismatch of source dynamic extents with static extents");
    }
  }

  _CCCL_TEMPLATE(class _OtherElementType, class _OtherExtents, class _OtherLayoutPolicy, class _OtherAccessor)
  _CCCL_REQUIRES((rank() == 0) //
                 _CCCL_AND __is_convertible_from<_OtherExtents, _OtherLayoutPolicy, _OtherAccessor> //
                   _CCCL_AND(!__is_implicit_convertible_from<_OtherExtents, _OtherLayoutPolicy, _OtherAccessor>))
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr mdspan(
    const mdspan<_OtherElementType, _OtherExtents, _OtherLayoutPolicy, _OtherAccessor>& __other)
      : __ptr_(__other.__ptr_)
      , __map_(__other.__map_)
      , __acc_(__other.__acc_)
  {
    static_assert(_CCCL_TRAIT(is_constructible, data_handle_type, const typename _OtherAccessor::data_handle_type&),
                  "mdspan: incompatible data_handle_type for mdspan construction");
    static_assert(_CCCL_TRAIT(is_constructible, extents_type, _OtherExtents),
                  "mdspan: incompatible extents for mdspan construction");
  }

  constexpr mdspan& operator=(const mdspan&) = default;
  constexpr mdspan& operator=(mdspan&&)      = default;

  //--------------------------------------------------------------------------------
  // [mdspan.mdspan.members], members

#if defined(_LIBCUDACXX_HAS_MULTIARG_OPERATOR_BRACKETS)
  _CCCL_TEMPLATE(class... _OtherIndexTypes)
  _CCCL_REQUIRES((sizeof...(_OtherIndexTypes) == rank())
                   _CCCL_AND _CCCL_FOLD_AND(_CCCL_TRAIT(is_convertible, _OtherIndexTypes, index_type))
                     _CCCL_AND _CCCL_FOLD_AND(_CCCL_TRAIT(is_nothrow_constructible, index_type, _OtherIndexTypes)))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference operator[](_OtherIndexTypes... __indices) const
  {
    // Note the standard layouts would also check this, but user provided ones may not, so we
    // check the precondition here
    _CCCL_ASSERT(__mdspan_detail::__is_multidimensional_index_in(extents(), __indices...),
                 "mdspan: operator[] out of bounds access");
    return __acc_.access(__ptr_, __map_(static_cast<index_type>(_CUDA_VSTD::move(__indices))...));
  }
#else
  _CCCL_TEMPLATE(class _OtherIndexType)
  _CCCL_REQUIRES((rank() == 1) _CCCL_AND _CCCL_TRAIT(is_convertible, _OtherIndexType, index_type)
                   _CCCL_AND _CCCL_TRAIT(is_nothrow_constructible, index_type, _OtherIndexType))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference operator[](_OtherIndexType __index) const
  {
    return __acc_.access(__ptr_, __map_(static_cast<index_type>(_CUDA_VSTD::move(__index))));
  }
#endif // _LIBCUDACXX_HAS_MULTIARG_OPERATOR_BRACKETS

  template <class _OtherIndexType, size_t... _Idxs>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto)
  __op_bracket(const array<_OtherIndexType, _Extents::rank()>& __indices, index_sequence<_Idxs...>) const noexcept
  {
    return __map_(__indices[_Idxs]...);
  }

  template <class _OtherIndexType, size_t... _Idxs>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto)
  __op_bracket(span<_OtherIndexType, _Extents::rank()> __indices, index_sequence<_Idxs...>) const noexcept
  {
    return __map_(__indices[_Idxs]...);
  }

  _CCCL_TEMPLATE(class _OtherIndexType)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, const _OtherIndexType&, index_type)
                   _CCCL_AND _CCCL_TRAIT(is_nothrow_constructible, index_type, const _OtherIndexType&))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference operator[](const array<_OtherIndexType, rank()>& __indices) const
  {
    return __acc_.access(__ptr_, __op_bracket(__indices, make_index_sequence<rank()>()));
  }

  _CCCL_TEMPLATE(class _OtherIndexType)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, const _OtherIndexType&, index_type)
                   _CCCL_AND _CCCL_TRAIT(is_nothrow_constructible, index_type, const _OtherIndexType&))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference operator[](span<_OtherIndexType, rank()> __indices) const
  {
    return __acc_.access(__ptr_, __op_bracket(__indices, make_index_sequence<rank()>()));
  }

  //! Nonstandard extension to no break our users too hard
  _CCCL_TEMPLATE(class... _OtherIndices)
  _CCCL_REQUIRES(_CCCL_FOLD_AND(_CCCL_TRAIT(is_convertible, const _OtherIndices&, index_type))
                   _CCCL_AND _CCCL_FOLD_AND(_CCCL_TRAIT(is_nothrow_constructible, index_type, const _OtherIndices&)))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference operator()(_OtherIndices... __indices) const
  {
    return __acc_.access(__ptr_, __map_(__indices...));
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __mul_overflow(size_t x, size_t y, size_t* res) noexcept
  {
    *res = x * y;
    return x && ((*res / x) != y);
  }

  template <size_t... _Idxs>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __check_size(index_sequence<_Idxs...>) const noexcept
  {
    size_t __prod = 1;
    return !_CCCL_FOLD_OR((__mul_overflow(__prod, __map_.extents().extent(_Idxs), &__prod)));
  }

  template <size_t... _Idxs>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr size_type __op_size(index_sequence<_Idxs...>) const noexcept
  {
    return _CCCL_FOLD_TIMES(size_type{1}, static_cast<size_type>(__map_.extents().extent(_Idxs)));
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr size_type size() const noexcept
  {
    // Could leave this as only checked in debug mode: semantically size() is never
    // guaranteed to be related to any accessible range
    _CCCL_ASSERT(__check_size(make_index_sequence<rank()>()), "mdspan: size() is not representable as size_type");
    return __op_size(make_index_sequence<rank()>());
  }

  template <size_t... _Idxs>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __op_empty(index_sequence<_Idxs...>) const noexcept
  {
    return _CCCL_FOLD_OR((__map_.extents().extent(_Idxs) == index_type{0}));
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool empty() const noexcept
  {
    return __op_empty(make_index_sequence<rank()>());
  }

  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr void swap(mdspan& __x, mdspan& __y) noexcept
  {
    swap(__x.__ptr_, __y.__ptr_);
    swap(__x.__map_, __y.__map_);
    swap(__x.__acc_, __y.__acc_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr const extents_type& extents() const noexcept
  {
    return __map_.extents();
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr const data_handle_type& data_handle() const noexcept
  {
    return __ptr_;
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr const mapping_type& mapping() const noexcept
  {
    return __map_;
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr const accessor_type& accessor() const noexcept
  {
    return __acc_;
  };

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_unique()
  {
    return mapping_type::is_always_unique();
  };
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_exhaustive()
  {
    return mapping_type::is_always_exhaustive();
  };
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_strided()
  {
    return mapping_type::is_always_strided();
  };

  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_unique() const
  {
    return __map_.is_unique();
  };
  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_exhaustive() const
  {
    return __map_.is_exhaustive();
  };
  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_strided() const
  {
    return __map_.is_strided();
  };
  _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type stride(rank_type __r) const
  {
    return __map_.stride(__r);
  };

private:
  _CCCL_NO_UNIQUE_ADDRESS data_handle_type __ptr_{};
  _CCCL_NO_UNIQUE_ADDRESS mapping_type __map_{};
  _CCCL_NO_UNIQUE_ADDRESS accessor_type __acc_{};

  template <class, class, class, class>
  friend class mdspan;
};

_CCCL_TEMPLATE(class _ElementType, class... _OtherIndexTypes)
_CCCL_REQUIRES((sizeof...(_OtherIndexTypes) > 0)
                 _CCCL_AND _CCCL_FOLD_AND(_CCCL_TRAIT(is_convertible, _OtherIndexTypes, size_t)))
_CCCL_HOST_DEVICE explicit mdspan(_ElementType*, _OtherIndexTypes...)
  -> mdspan<_ElementType, extents<size_t, __maybe_static_ext<_OtherIndexTypes>...>>;

_CCCL_TEMPLATE(class _Pointer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_pointer, remove_reference_t<_Pointer>))
_CCCL_HOST_DEVICE mdspan(_Pointer&&) -> mdspan<remove_pointer_t<remove_reference_t<_Pointer>>, extents<size_t>>;

_CCCL_TEMPLATE(class _CArray)
_CCCL_REQUIRES(_CCCL_TRAIT(is_array, _CArray) _CCCL_AND(rank_v<_CArray> == 1))
_CCCL_HOST_DEVICE mdspan(_CArray&) -> mdspan<remove_all_extents_t<_CArray>, extents<size_t, extent_v<_CArray, 0>>>;

template <class _ElementType, class _OtherIndexType, size_t _Size>
_CCCL_HOST_DEVICE mdspan(_ElementType*, const array<_OtherIndexType, _Size>&)
  -> mdspan<_ElementType, dextents<size_t, _Size>>;

template <class _ElementType, class _OtherIndexType, size_t _Size>
_CCCL_HOST_DEVICE mdspan(_ElementType*, span<_OtherIndexType, _Size>) -> mdspan<_ElementType, dextents<size_t, _Size>>;

// This one is necessary because all the constructors take `data_handle_type`s, not
// `_ElementType*`s, and `data_handle_type` is taken from `accessor_type::data_handle_type`, which
// seems to throw off automatic deduction guides.
template <class _ElementType, class _OtherIndexType, size_t... _ExtentsPack>
_CCCL_HOST_DEVICE mdspan(_ElementType*, const extents<_OtherIndexType, _ExtentsPack...>&)
  -> mdspan<_ElementType, extents<_OtherIndexType, _ExtentsPack...>>;

template <class _ElementType, class _MappingType>
_CCCL_HOST_DEVICE mdspan(_ElementType*, const _MappingType&)
  -> mdspan<_ElementType, typename _MappingType::extents_type, typename _MappingType::layout_type>;

template <class _MappingType, class _AccessorType>
_CCCL_HOST_DEVICE mdspan(const typename _AccessorType::data_handle_type, const _MappingType&, const _AccessorType&)
  -> mdspan<typename _AccessorType::element_type,
            typename _MappingType::extents_type,
            typename _MappingType::layout_type,
            _AccessorType>;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MDSPAN_MDSPAN_H
