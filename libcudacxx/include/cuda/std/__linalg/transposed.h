/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software. //
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#ifndef _LIBCUDACXX___LINALG_TRANSPOSED_HPP
#define _LIBCUDACXX___LINALG_TRANSPOSED_HPP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/version>

#if defined(__cccl_lib_mdspan) && _CCCL_STD_VER >= 2017

#  include <cuda/std/__cccl/assert.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/array>
#  include <cuda/std/mdspan>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace linalg
{

namespace __detail
{
// This struct helps us impose the rank constraint on the __type alias itself.
_CCCL_TEMPLATE(class _Extents)
_CCCL_REQUIRES((_Extents::rank() == 2))
struct __transpose_extents_t_impl
{
  using __type =
    _CUDA_VSTD::extents<typename _Extents::index_type, _Extents::static_extent(1), _Extents::static_extent(0)>;
};

template <class _Extents>
using __transpose_extents_t = typename __transpose_extents_t_impl<_Extents>::__type;

_CCCL_TEMPLATE(class _Extents)
_CCCL_REQUIRES((_Extents::rank() == 2))
_LIBCUDACXX_HIDE_FROM_ABI __transpose_extents_t<_Extents> __transpose_extents(const _Extents& __e)
{
  static_assert(
    _CUDA_VSTD::is_same_v<typename __transpose_extents_t<_Extents>::index_type, typename _Extents::index_type>,
    "Please fix __transpose_extents_t to account  for P2553, which adds __a template parameter SizeType to extents.");
  constexpr size_t __ext0 = _Extents::static_extent(0);
  constexpr size_t __ext1 = _Extents::static_extent(1);
  if constexpr (__ext0 == _CUDA_VSTD::dynamic_extent)
  {
    if constexpr (__ext1 == _CUDA_VSTD::dynamic_extent)
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
    if constexpr (__ext1 == _CUDA_VSTD::dynamic_extent)
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

  public:
    using extents_type = _Extents;
    using index_type   = typename extents_type::index_type;
    using size_type    = typename extents_type::size_type;
    using rank_type    = typename extents_type::rank_type;
    using layout_type  = layout_transpose;

    _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit mapping(const __nested_mapping_type& __map)
        : __nested_mapping_(__map)
        , __extents_(__detail::__transpose_extents(__map.extents()))
    {}

    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr const extents_type& extents() const noexcept
    {
      return __extents_;
    }

    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type required_span_size() const
#  if !_CCCL_COMPILER(GCC, <=, 9) && !_CCCL_COMPILER(CLANG, <=, 9)
      noexcept(noexcept(__nested_mapping_.required_span_size()))
#  endif
    {
      return __nested_mapping_.required_span_size();
    }

    _CCCL_TEMPLATE(class _IndexType0, class _IndexType1)
    _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _IndexType0, index_type)
                     _CCCL_AND _CCCL_TRAIT(is_convertible, _IndexType1, index_type))
    _LIBCUDACXX_HIDE_FROM_ABI index_type operator()(_IndexType0 __i, _IndexType1 __j) const
    {
      return __nested_mapping_(__j, __i);
    }

    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI const __nested_mapping_type& nested_mapping() const
    {
      return __nested_mapping_;
    }

    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_unique() noexcept
    {
      return __nested_mapping_type::is_always_unique();
    }

    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_exhaustive() noexcept
    {
      return __nested_mapping_type::is_always_contiguous();
    }

    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_strided() noexcept
    {
      return __nested_mapping_type::is_always_strided();
    }

    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_unique() const
    {
      return __nested_mapping_.is_unique();
    }

    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_exhaustive() const
    {
      return __nested_mapping_.is_exhaustive();
    }

    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_strided() const
    {
      return __nested_mapping_.is_strided();
    }

    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type stride(size_t __r) const
    {
      _CCCL_ASSERT(this->is_strided(), "layout must be strided");
      _CCCL_ASSERT(__r < extents_type::rank(), "rank must be less than extents rank");
      return __nested_mapping_.stride(__r == 0 ? 1 : 0);
    }

    template <class _OtherExtents>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
    operator==(const mapping& __lhs, const mapping<_OtherExtents>& __rhs) noexcept
    {
      return __lhs.__nested_mapping_ == __rhs.__nested_mapping_;
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

  _LIBCUDACXX_HIDE_FROM_ABI static __accessor_type __accessor(const _Accessor& __a)
  {
    return __accessor_type(__a);
  }
};

template <class _ElementType>
struct __transposed_element_accessor<_ElementType, _CUDA_VSTD::default_accessor<_ElementType>>
{
  using __element_type  = _ElementType;
  using __accessor_type = _CUDA_VSTD::default_accessor<__element_type>;

  _LIBCUDACXX_HIDE_FROM_ABI static __accessor_type __accessor(const _CUDA_VSTD::default_accessor<_ElementType>& __a)
  {
    return __accessor_type(__a);
  }
};

template <class _Layout>
struct __transposed_layout
{
  using __layout_type = layout_transpose<_Layout>;

  template <class __OriginalMapping>
  _LIBCUDACXX_HIDE_FROM_ABI static auto __mapping(const __OriginalMapping& __orig_map)
  {
    using __extents_type        = __transpose_extents_t<typename __OriginalMapping::__extents_type>;
    using __return_mapping_type = typename __layout_type::template __mapping<__extents_type>;
    return __return_mapping_type{__orig_map};
  }
};

template <>
struct __transposed_layout<_CUDA_VSTD::layout_left>
{
  using __layout_type = _CUDA_VSTD::layout_right;

  template <class _OriginalExtents>
  _LIBCUDACXX_HIDE_FROM_ABI static auto
  __mapping(const typename _CUDA_VSTD::layout_left::template mapping<_OriginalExtents>& __orig_map)
  {
    using __original_mapping_type = typename _CUDA_VSTD::layout_left::template mapping<_OriginalExtents>;
    using __extents_type          = __transpose_extents_t<typename __original_mapping_type::extents_type>;
    using __return_mapping_type   = typename __layout_type::template mapping<__extents_type>;
    return __return_mapping_type{__transpose_extents(__orig_map.extents())};
  }
};

template <>
struct __transposed_layout<_CUDA_VSTD::layout_right>
{
  using __layout_type = _CUDA_VSTD::layout_left;

  template <class _OriginalExtents>
  _LIBCUDACXX_HIDE_FROM_ABI static auto
  __mapping(const typename _CUDA_VSTD::layout_right::template mapping<_OriginalExtents>& __orig_map)
  {
    using __original_mapping_type = typename _CUDA_VSTD::layout_right::template mapping<_OriginalExtents>;
    using __extents_type          = __transpose_extents_t<typename __original_mapping_type::extents_type>;
    using __return_mapping_type   = typename __layout_type::template mapping<__extents_type>;
    return __return_mapping_type{__transpose_extents(__orig_map.extents())};
  }
};

template <>
struct __transposed_layout<_CUDA_VSTD::layout_stride>
{
  using __layout_type = _CUDA_VSTD::layout_stride;

  template <class _OriginalExtents>
  _LIBCUDACXX_HIDE_FROM_ABI static auto
  __mapping(const typename _CUDA_VSTD::layout_stride::template mapping<_OriginalExtents>& __orig_map)
  {
    using __original_mapping_type = typename _CUDA_VSTD::layout_stride::template mapping<_OriginalExtents>;
    using __original_extents_type = typename __original_mapping_type::extents_type;
    using __extents_type          = __transpose_extents_t<__original_extents_type>;
    using __return_mapping_type   = typename __layout_type::template mapping<__extents_type>;
    return __return_mapping_type{
      __transpose_extents(__orig_map.extents()),
      _CUDA_VSTD::array<typename __extents_type::index_type, _OriginalExtents::rank() /* __orig_map.rank() */>{
        __orig_map.stride(1), __orig_map.stride(0)}};
  }
};

// TODO add support for padded layouts

#  if 0

template <class _StorageOrder>
using __opposite_storage_t =
  _CUDA_VSTD::conditional_t<_CUDA_VSTD::is_same_v<_StorageOrder, column_major_t>, row_major_t, column_major_t>;

template <class _StorageOrder>
struct __transposed_layout<layout_blas_general<_StorageOrder>>
{
  using __layout_type = layout_blas_general<__opposite_storage_t<_StorageOrder>>;

  template <class _OriginalExtents>
  _LIBCUDACXX_HIDE_FROM_ABI static auto
  __mapping(const typename layout_blas_general<_StorageOrder>::template __mapping<_OriginalExtents>& __orig_map)
  {
    using __original_mapping_type = typename layout_blas_general<_StorageOrder>::template __mapping<_OriginalExtents>;
    using __extents_type          = __transpose_extents_t<typename __original_mapping_type::__extents_type>;
    using __return_mapping_type   = typename __layout_type::template __mapping<__extents_type>;
    const auto whichStride =
      _CUDA_VSTD::is_same_v<_StorageOrder, column_major_t> ? __orig_map.stride(1) : __orig_map.stride(0);
    return __return_mapping_type{__transpose_extents(__orig_map.extents()), whichStride};
  }
};

template <class Triangle>
using opposite_triangle_t =
  _CUDA_VSTD::conditional_t<_CUDA_VSTD::is_same_v<Triangle, upper_triangle_t>, lower_triangle_t, upper_triangle_t>;

template <class Triangle, class _StorageOrder>
struct __transposed_layout<layout_blas_packed<Triangle, _StorageOrder>>
{
  using __layout_type = layout_blas_packed<opposite_triangle_t<Triangle>, __opposite_storage_t<_StorageOrder>>;

  template <class _OriginalExtents>
  _LIBCUDACXX_HIDE_FROM_ABI static auto
  __mapping(const typename layout_blas_packed<Triangle, _StorageOrder>::template __mapping<_OriginalExtents>& __orig_map)
  {
    using __original_mapping_type =
      typename layout_blas_packed<Triangle, _StorageOrder>::template __mapping<_OriginalExtents>;
    using __extents_type        = __transpose_extents_t<typename __original_mapping_type::__extents_type>;
    using __return_mapping_type = typename __layout_type::template __mapping<__extents_type>;
    return __return_mapping_type{__transpose_extents(__orig_map.extents())};
  }
};
#  endif

template <class _NestedLayout>
struct __transposed_layout<layout_transpose<_NestedLayout>>
{
  using __layout_type = _NestedLayout;
};

} // namespace __detail

template <class _ElementType, class _Extents, class _Layout, class _Accessor>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI auto
transposed(_CUDA_VSTD::mdspan<_ElementType, _Extents, _Layout, _Accessor> __a)
{
  using __element_type  = typename __detail::__transposed_element_accessor<_ElementType, _Accessor>::__element_type;
  using __layout_type   = typename __detail::__transposed_layout<_Layout>::__layout_type;
  using __accessor_type = typename __detail::__transposed_element_accessor<_ElementType, _Accessor>::__accessor_type;
  auto __mapping        = __detail::__transposed_layout<_Layout>::__mapping(__a.mapping());
  auto __accessor       = __detail::__transposed_element_accessor<_ElementType, _Accessor>::__accessor(__a.accessor());
  return _CUDA_VSTD::mdspan<__element_type, typename decltype(__mapping)::extents_type, __layout_type, __accessor_type>{
    __a.data_handle(), __mapping, __accessor};
}

} // end namespace linalg

_LIBCUDACXX_END_NAMESPACE_STD

#endif // defined(__cccl_lib_mdspan) && _CCCL_STD_VER >= 2017
#endif // _LIBCUDACXX___LINALG_TRANSPOSED_HPP