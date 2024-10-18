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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_SCALED_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_SCALED_HPP_

#include <experimental/mdspan>

namespace std
{
namespace experimental
{
inline namespace __p1673_version_0
{
namespace linalg
{

template <class ScalingFactor, class Accessor>
class accessor_scaled
{
public:
  using reference =
    scaled_scalar<ScalingFactor, typename Accessor::reference, std::remove_cv_t<typename Accessor::element_type>>;
  using element_type     = std::add_const_t<typename reference::value_type>;
  using data_handle_type = typename Accessor::data_handle_type;
  using offset_policy    = accessor_scaled<ScalingFactor, typename Accessor::offset_policy>;

  accessor_scaled(ScalingFactor scaling_factor, Accessor accessor)
      : scaling_factor_(std::move(scaling_factor))
      , accessor_(accessor)
  {}

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType,
    /* requires */ (std::is_convertible_v<typename default_accessor<OtherElementType>::element_type (*)[],
                                          typename Accessor::element_type (*)[]>) )
  accessor_scaled(ScalingFactor scaling_factor, default_accessor<OtherElementType> accessor)
      : scaling_factor_(std::move(scaling_factor))
      , accessor_(accessor)
  {}

  reference access(data_handle_type p, ::std::size_t i) const noexcept
  {
    return reference(scaling_factor_, accessor_.access(p, i));
  }

  typename offset_policy::data_handle_type offset(data_handle_type p, ::std::size_t i) const noexcept
  {
    return accessor_.offset(p, i);
  }

  Accessor nested_accessor() const
  {
    return accessor_;
  }

  ScalingFactor scaling_factor() const
  {
    return scaling_factor_;
  }

private:
  ScalingFactor scaling_factor_;
  Accessor accessor_;
};

namespace impl
{

template <class ScalingFactor, class Accessor>
using scaled_element_type = std::add_const_t<typename accessor_scaled<ScalingFactor, Accessor>::element_type>;

} // namespace impl

template <class ScalingFactor, class ElementType, class Extents, class Layout, class Accessor>
mdspan<impl::scaled_element_type<ScalingFactor, Accessor>, Extents, Layout, accessor_scaled<ScalingFactor, Accessor>>
scaled(ScalingFactor scaling_factor, mdspan<ElementType, Extents, Layout, Accessor> x)
{
  using acc_type = accessor_scaled<ScalingFactor, Accessor>;
  return {x.data_handle(), x.mapping(), acc_type{scaling_factor, x.accessor()}};
}

} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std

#endif // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_SCALED_HPP_
