/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_LAYOUT_BLAS_GENERAL_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_LAYOUT_BLAS_GENERAL_HPP_

#include "layout_tags.hpp"
#include "maybe_static_size.hpp"

namespace std
{
namespace experimental
{
inline namespace __p1673_version_0
{
namespace linalg
{

namespace __layout_blas_general_impl
{

template <class BaseLayout, ::std::size_t StaticLDA>
class __layout_blas_impl
{
private:
  _MDSPAN_NO_UNIQUE_ADDRESS BaseLayout _base_layout;

public: // but not really
  using __lda_t                           = detail::__maybe_static_extent<StaticLDA>;
  _MDSPAN_NO_UNIQUE_ADDRESS __lda_t __lda = {};

private:
  using __extents_type = decltype(std::declval<BaseLayout const&>().extents());

  template <class, ::std::size_t>
  friend class __layout_blas_impl;

public:
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr __layout_blas_impl() noexcept                          = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr __layout_blas_impl(__layout_blas_impl const&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr __layout_blas_impl(__layout_blas_impl&&) noexcept      = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED __layout_blas_impl&
  operator=(__layout_blas_impl const&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED __layout_blas_impl&
  operator=(__layout_blas_impl&&) noexcept               = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED ~__layout_blas_impl() = default;

  MDSPAN_INLINE_FUNCTION
  constexpr explicit __layout_blas_impl(__extents_type const& exts) noexcept
      : _base_layout(exts)
      , __lda(1)
  {}

  MDSPAN_FUNCTION_REQUIRES(
    (MDSPAN_INLINE_FUNCTION constexpr),
    __layout_blas_impl,
    (__extents_type const& exts, ::std::size_t lda),
    noexcept,
    /* requires */ (!__lda_t::is_static))
      : _base_layout(exts)
      , __lda(lda)
  {}

  // TODO noexcept specification
  // TODO throw if rhs is dynamic LDA and doesn't match static lhs
  MDSPAN_TEMPLATE_REQUIRES(
    class OtherExtents,
    ::std::size_t OtherLDA, /* requires */
    (_MDSPAN_TRAIT(is_convertible, OtherExtents, __extents_type)
     && (!__layout_blas_impl<OtherExtents, OtherLDA>::__lda_t::is_static || !__lda_t::is_static
         || __lda_t::value_static == OtherLDA)))
  MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
  __layout_blas_impl(__layout_blas_impl<OtherExtents, OtherLDA> const& other) // NOLINT(google-explicit-constructor)
      : _base_layout(other.extents())
      , __lda(other.__lda)
  {}

  // TODO noexcept specification
  // TODO throw if rhs is dynamic LDA and doesn't match static lhs
  MDSPAN_TEMPLATE_REQUIRES(
    class OtherExtents,
    ::std::size_t OtherLDA,
    /* requires */
    (_MDSPAN_TRAIT(is_convertible, OtherExtents, __extents_type)
     && (!__layout_blas_impl<OtherExtents, OtherLDA>::__lda_t::is_static || !__lda_t::is_static
         || __lda_t::value_static == OtherLDA)))
  MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 __layout_blas_impl&
  operator=(__layout_blas_impl<OtherExtents, OtherLDA> const& other)
  {
    this->_extents = other.extents();
    this->__lda    = other.__lda.value;
    return *this;
  }

  template <class... Integral>
  MDSPAN_FORCE_INLINE_FUNCTION constexpr ::std::size_t operator()(Integral... idxs) const noexcept
  {
    return __lda.value * _base_layout(idxs...);
  }

  MDSPAN_INLINE_FUNCTION constexpr bool is_unique() const noexcept
  {
    return true;
  }
  MDSPAN_INLINE_FUNCTION constexpr bool is_contiguous() const noexcept
  {
    return __lda.value == 1;
  }
  MDSPAN_INLINE_FUNCTION constexpr bool is_strided() const noexcept
  {
    return true;
  }

  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept
  {
    return true;
  }
  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_contiguous() noexcept
  {
    return __lda_t::value_static == 1;
  }
  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept
  {
    return true;
  }

  MDSPAN_INLINE_FUNCTION constexpr __extents_type extents() const noexcept
  {
    return _base_layout.extents();
  }

  MDSPAN_INLINE_FUNCTION
  constexpr typename __extents_type::size_type required_span_size() const noexcept
  {
    return _base_layout.required_span_size() * __lda.value;
  }

  MDSPAN_INLINE_FUNCTION
  constexpr typename __extents_type::size_type stride(size_t r) const noexcept
  {
    return _base_layout.stride(r) * __lda.value;
  }

  template <class OtherExtents, typename __extents_type::size_type OtherLDA>
  MDSPAN_INLINE_FUNCTION friend constexpr bool
  operator==(__layout_blas_impl const& a, __layout_blas_impl<OtherExtents, OtherLDA> const& b) noexcept
  {
    return a.extents() == b.extents() && a.__lda == b.__lda;
  }

  template <class OtherExtents, typename __extents_type::size_type OtherLDA>
  MDSPAN_INLINE_FUNCTION friend constexpr bool
  operator!=(__layout_blas_impl const& a, __layout_blas_impl<OtherExtents, OtherLDA> const& b) noexcept
  {
    return a.extents() != b.extents() || a.__lda != b.__lda;
  }

  // Needed to work with subspan()
  template <size_t N>
  struct __static_stride_workaround
  {
    static constexpr typename __extents_type::size_type value =
      __lda_t::is_static
        ? (BaseLayout::template __static_stride_workaround<N>::value == dynamic_extent
             ? dynamic_extent
             : (__lda_t::value_static * BaseLayout::template __static_stride_workaround<N>::value))
        : dynamic_extent;
  };
};

} // end namespace __layout_blas_general_impl

template <class StorageOrder>
class layout_blas_general;

template <>
class layout_blas_general<column_major_t>
{
  template <class Extents>
  using mapping = __layout_blas_general_impl::__layout_blas_impl<layout_left, dynamic_extent>;
};

template <>
class layout_blas_general<row_major_t>
{
  template <class Extents>
  using mapping = __layout_blas_general_impl::__layout_blas_impl<layout_right, dynamic_extent>;
};

} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std

#endif // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_LAYOUT_BLAS_GENERAL_HPP_
