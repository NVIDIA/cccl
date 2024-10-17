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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATED_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATED_HPP_

#include <experimental/mdspan>

namespace std
{
namespace experimental
{
inline namespace __p1673_version_0
{
namespace linalg
{

template <class Accessor>
class accessor_conjugate;

namespace impl
{
template <class Accessor, bool is_arith = std::is_arithmetic_v<std::remove_cv_t<typename Accessor::element_type>>>
struct accessor_conjugate_aliases
{};

template <class Accessor>
struct accessor_conjugate_aliases<Accessor, true>
{
  using reference        = typename Accessor::reference;
  using element_type     = std::add_const_t<typename Accessor::element_type>;
  using data_handle_type = typename Accessor::data_handle_type;
  using offset_policy    = typename Accessor::offset_policy;
};

template <class Accessor>
struct accessor_conjugate_aliases<Accessor, false>
{
private:
  using accessor_value_type = std::remove_cv_t<typename Accessor::element_type>;

public:
  using reference        = conjugated_scalar<typename Accessor::reference, accessor_value_type>;
  using element_type     = std::add_const_t<typename reference::value_type>;
  using data_handle_type = typename Accessor::data_handle_type;
  using offset_policy    = accessor_conjugate<typename Accessor::offset_policy>;
};

} // namespace impl

template <class Accessor>
class accessor_conjugate
{
private:
  Accessor accessor_;
  using aliases = impl::accessor_conjugate_aliases<Accessor>;

public:
  using reference        = typename aliases::reference;
  using element_type     = typename aliases::element_type;
  using data_handle_type = typename aliases::data_handle_type;
  using offset_policy    = typename aliases::offset_policy;

  accessor_conjugate(Accessor accessor)
      : accessor_(accessor)
  {}

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType,
    /* requires */ (std::is_convertible_v<typename default_accessor<OtherElementType>::element_type (*)[],
                                          typename Accessor::element_type (*)[]>) )
  accessor_conjugate(default_accessor<OtherElementType> accessor)
      : accessor_(accessor)
  {}

  reference access(data_handle_type p, ::std::size_t i) const noexcept(noexcept(reference(accessor_.access(p, i))))
  {
    return reference(accessor_.access(p, i));
  }

  typename offset_policy::data_handle_type offset(data_handle_type p, ::std::size_t i) const
    noexcept(noexcept(accessor_.offset(p, i)))
  {
    return accessor_.offset(p, i);
  }

  Accessor nested_accessor() const
  {
    return accessor_;
  }
};

template <class ElementType, class Extents, class Layout, class Accessor>
auto conjugated(mdspan<ElementType, Extents, Layout, Accessor> a)
{
  if constexpr (std::is_arithmetic_v<std::remove_cv_t<ElementType>>)
  {
    return mdspan<ElementType, Extents, Layout, Accessor>(a.data_handle(), a.mapping(), a.accessor());
  }
  else
  {
    using return_element_type  = typename accessor_conjugate<Accessor>::element_type;
    using return_accessor_type = accessor_conjugate<Accessor>;
    return mdspan<return_element_type, Extents, Layout, return_accessor_type>(
      a.data_handle(), a.mapping(), return_accessor_type(a.accessor()));
  }
}

// Conjugation is self-annihilating
template <class ElementType, class Extents, class Layout, class NestedAccessor>
auto conjugated(mdspan<ElementType, Extents, Layout, accessor_conjugate<NestedAccessor>> a)
{
  using return_element_type  = typename NestedAccessor::element_type;
  using return_accessor_type = NestedAccessor;
  return mdspan<return_element_type, Extents, Layout, return_accessor_type>(
    a.data_handle(), a.mapping(), a.nested_accessor());
}

} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std

#endif // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATED_HPP_
