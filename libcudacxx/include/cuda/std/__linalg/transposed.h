//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
// ************************************************************************
//@HEADER

#ifndef _CUDA_STD___LINALG_TRANSPOSED_H
#define _CUDA_STD___LINALG_TRANSPOSED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/array>
#include <cuda/std/mdspan>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace linalg
{

namespace __detail
{
// This struct helps us impose the rank constraint on the __type alias itself.
_CCCL_TEMPLATE(class _Extents)
_CCCL_REQUIRES((_Extents::rank() == 2))
struct __transpose_extents_t_impl
{
  using __type = extents<typename _Extents::index_type, _Extents::static_extent(1), _Extents::static_extent(0)>;
};

template <class _Extents>
using __transpose_extents_t = typename __transpose_extents_t_impl<_Extents>::__type;

_CCCL_TEMPLATE(class _Extents)
_CCCL_REQUIRES((_Extents::rank() == 2))
_CCCL_API constexpr __transpose_extents_t<_Extents> __transpose_extents(const _Extents& __e)
{
  static_assert(is_same_v<typename __transpose_extents_t<_Extents>::index_type, typename _Extents::index_type>,
                "Please fix __transpose_extents_t to account for P2553, which adds a template parameter SizeType to "
                "extents.");
  constexpr size_t __ext0 = _Extents::static_extent(0);
  constexpr size_t __ext1 = _Extents::static_extent(1);
  if constexpr (__ext0 == dynamic_extent)
  {
    if constexpr (__ext1 == dynamic_extent)
    {
      return __transpose_extents_t<_Extents>{__e.extent(1), __e.extent(0)};
    }
    else
    {
      return __transpose_extents_t<_Extents>{/* __e.extent(1), */ __e.extent(0)};
    }
  }
  else
  {
    if constexpr (__ext1 == dynamic_extent)
    {
      return __transpose_extents_t<_Extents>{__e.extent(1) /* , __e.extent(0) */};
    }
    else
    {
      return __transpose_extents_t<_Extents>{}; // all extents are static
    }
  }
  _CCCL_UNREACHABLE(); // GCC9 workaround
}

} // namespace __detail

template <class _Layout>
class layout_transpose
{
public:
  using nested_layout_type = _Layout;

  template <class _Extents>
  struct mapping
  {
  private:
    using __nested_mapping_type = typename _Layout::template mapping<__detail::__transpose_extents_t<_Extents>>;

    static constexpr bool __required_span_size_noexcept = noexcept(__nested_mapping_type{}.required_span_size());

    static constexpr bool __is_nested_unique_noexcept = noexcept(__nested_mapping_type{}.is_unique());

    static constexpr bool __is_exhaustive_noexcept = noexcept(__nested_mapping_type{}.is_exhaustive());

    static constexpr bool __is_strided_noexcept = noexcept(__nested_mapping_type{}.is_strided());

  public:
    using extents_type = _Extents;
    using index_type   = typename extents_type::index_type;
    using size_type    = typename extents_type::size_type;
    using rank_type    = typename extents_type::rank_type;
    using layout_type  = layout_transpose;

    _CCCL_API constexpr explicit mapping(const __nested_mapping_type& __map)
        : __nested_mapping_(__map)
        , __extents_(__detail::__transpose_extents(__map.extents()))
    {}

    [[nodiscard]] _CCCL_API constexpr const extents_type& extents() const noexcept
    {
      return __extents_;
    }

    [[nodiscard]] _CCCL_API constexpr index_type required_span_size() const noexcept(__required_span_size_noexcept)
    {
      return __nested_mapping_.required_span_size();
    }

    _CCCL_TEMPLATE(class _IndexType0, class _IndexType1)
    _CCCL_REQUIRES(is_convertible_v<_IndexType0, index_type> _CCCL_AND is_convertible_v<_IndexType1, index_type>)
    _CCCL_API constexpr index_type operator()(_IndexType0 __i, _IndexType1 __j) const
    {
      return __nested_mapping_(__j, __i);
    }

    [[nodiscard]] _CCCL_API constexpr const __nested_mapping_type& nested_mapping() const noexcept
    {
      return __nested_mapping_;
    }

    [[nodiscard]] _CCCL_API static constexpr bool is_always_unique() noexcept
    {
      return __nested_mapping_type::is_always_unique();
    }

    [[nodiscard]] _CCCL_API static constexpr bool is_always_exhaustive() noexcept
    {
      return __nested_mapping_type::is_always_exhaustive();
    }

    [[nodiscard]] _CCCL_API static constexpr bool is_always_strided() noexcept
    {
      return __nested_mapping_type::is_always_strided();
    }

    [[nodiscard]] _CCCL_API constexpr bool is_unique() const noexcept(__is_nested_unique_noexcept)
    {
      return __nested_mapping_.is_unique();
    }

    [[nodiscard]] _CCCL_API constexpr bool is_exhaustive() const noexcept(__is_exhaustive_noexcept)
    {
      return __nested_mapping_.is_exhaustive();
    }

    [[nodiscard]] _CCCL_API constexpr bool is_strided() const noexcept(__is_strided_noexcept)
    {
      return __nested_mapping_.is_strided();
    }

    [[nodiscard]] _CCCL_API constexpr index_type stride(size_t __r) const
    {
      _CCCL_ASSERT(this->is_strided(), "layout must be strided");
      _CCCL_ASSERT(__r < extents_type::rank(), "rank must be less than extents rank");
      return __nested_mapping_.stride(__r == 0 ? 1 : 0);
    }

    template <class _OtherExtents>
    _CCCL_API friend constexpr bool operator==(const mapping& __lhs, const mapping<_OtherExtents>& __rhs) noexcept
    {
      return __lhs.__nested_mapping_ == __rhs.__nested_mapping_;
    }

    template <class _OtherExtents>
    _CCCL_API friend constexpr bool operator!=(const mapping& __lhs, const mapping<_OtherExtents>& __rhs) noexcept
    {
      return __lhs.__nested_mapping_ != __rhs.__nested_mapping_;
    }

  private:
    __nested_mapping_type __nested_mapping_;
    extents_type __extents_;
  };
};

namespace __detail
{

template <class _ElementType, class _Accessor>
struct __transposed_element_accessor
{
  using __element_type  = _ElementType;
  using __accessor_type = _Accessor;

  _CCCL_API static constexpr __accessor_type __accessor(const _Accessor& __a)
  {
    return __accessor_type(__a);
  }
};

template <class _ElementType>
struct __transposed_element_accessor<_ElementType, default_accessor<_ElementType>>
{
  using __element_type  = _ElementType;
  using __accessor_type = default_accessor<__element_type>;

  _CCCL_API static constexpr __accessor_type __accessor(const default_accessor<_ElementType>& __a)
  {
    return __accessor_type(__a);
  }
};

template <class _Layout>
struct __transposed_layout
{
  using __layout_type = layout_transpose<_Layout>;

  template <class __OriginalMapping>
  _CCCL_API static constexpr auto __mapping(const __OriginalMapping& __orig_map)
  {
    using __extents_type        = __transpose_extents_t<typename __OriginalMapping::__extents_type>;
    using __return_mapping_type = typename __layout_type::template __mapping<__extents_type>;
    return __return_mapping_type{__orig_map};
  }
};

template <>
struct __transposed_layout<layout_left>
{
  using __layout_type = layout_right;

  template <class _OriginalExtents>
  _CCCL_API static constexpr auto __mapping(const typename layout_left::template mapping<_OriginalExtents>& __orig_map)
  {
    using __original_mapping_type = typename layout_left::template mapping<_OriginalExtents>;
    using __extents_type          = __transpose_extents_t<typename __original_mapping_type::extents_type>;
    using __return_mapping_type   = typename __layout_type::template mapping<__extents_type>;
    return __return_mapping_type{__transpose_extents(__orig_map.extents())};
  }
};

template <>
struct __transposed_layout<layout_right>
{
  using __layout_type = layout_left;

  template <class _OriginalExtents>
  _CCCL_API static constexpr auto __mapping(const typename layout_right::template mapping<_OriginalExtents>& __orig_map)
  {
    using __original_mapping_type = typename layout_right::template mapping<_OriginalExtents>;
    using __extents_type          = __transpose_extents_t<typename __original_mapping_type::extents_type>;
    using __return_mapping_type   = typename __layout_type::template mapping<__extents_type>;
    return __return_mapping_type{__transpose_extents(__orig_map.extents())};
  }
};

template <>
struct __transposed_layout<layout_stride>
{
  using __layout_type = layout_stride;

  template <class _OriginalExtents>
  _CCCL_API static constexpr auto __mapping(const typename layout_stride::template mapping<_OriginalExtents>& __orig_map)
  {
    using __original_mapping_type = typename layout_stride::template mapping<_OriginalExtents>;
    using __original_extents_type = typename __original_mapping_type::extents_type;
    using __extents_type          = __transpose_extents_t<__original_extents_type>;
    using __return_mapping_type   = typename __layout_type::template mapping<__extents_type>;
    return __return_mapping_type{
      __transpose_extents(__orig_map.extents()),
      array<typename __extents_type::index_type, _OriginalExtents::rank() /* __orig_map.rank() */>{
        __orig_map.stride(1), __orig_map.stride(0)}};
  }
};

// TODO add support for padded layouts

template <class _NestedLayout>
struct __transposed_layout<layout_transpose<_NestedLayout>>
{
  using __layout_type = _NestedLayout;
};

} // namespace __detail

template <class _ElementType, class _Extents, class _Layout, class _Accessor>
[[nodiscard]] _CCCL_API constexpr auto transposed(mdspan<_ElementType, _Extents, _Layout, _Accessor> __a)
{
  using __element_type  = typename __detail::__transposed_element_accessor<_ElementType, _Accessor>::__element_type;
  using __layout_type   = typename __detail::__transposed_layout<_Layout>::__layout_type;
  using __accessor_type = typename __detail::__transposed_element_accessor<_ElementType, _Accessor>::__accessor_type;
  auto __mapping        = __detail::__transposed_layout<_Layout>::__mapping(__a.mapping());
  auto __accessor       = __detail::__transposed_element_accessor<_ElementType, _Accessor>::__accessor(__a.accessor());
  return mdspan<__element_type, typename decltype(__mapping)::extents_type, __layout_type, __accessor_type>{
    __a.data_handle(), __mapping, __accessor};
}

} // end namespace linalg

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___LINALG_TRANSPOSED_HPP
