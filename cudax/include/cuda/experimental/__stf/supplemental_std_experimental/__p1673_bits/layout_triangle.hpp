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

#ifndef LINALG_INCLUDE_EXPERIMENTAL_BITS_LAYOUT_TRIANGLE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL_BITS_LAYOUT_TRIANGLE_HPP_

#include "layout_tags.hpp"

#include <cstdint>
#include <type_traits>

namespace std { namespace experimental { inline namespace __p1673_version_0 { namespace linalg {

namespace __triangular_layouts_impl {

template <class, class, class, class, class, class>
struct __lower_triangle_layout_impl;

// FIXME work-around for #4.
#if 0

// lower triangular offsets are triangular numbers (n*(n+1)/2)
template <
  ptrdiff_t ExtLast, ptrdiff_t... Exts, class BaseMap, class LastTwoMap,
  size_t... ExtIdxs, size_t... ExtMinus2Idxs
>
struct __lower_triangle_layout_impl<
  std::experimental::extents<Exts..., ExtLast, ExtLast>,
  BaseMap, LastTwoMap,
  std::integer_sequence<size_t, ExtIdxs...>,
  std::integer_sequence<size_t, ExtMinus2Idxs...>
> {

private:

  static constexpr auto __rank = sizeof...(Exts) + 2;

  _MDSPAN_NO_UNIQUE_ADDRESS LastTwoMap _trimap;
  _MDSPAN_NO_UNIQUE_ADDRESS BaseMap _base_map;

public:


  template <class... Integral>
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr ptrdiff_t operator()(Integral... idxs) const noexcept {
    auto base_val = _base_map(
      [&](size_t N) {
        _MDSPAN_FOLD_PLUS_RIGHT(((ExtIdxs == N) ? idx : 0), /* + ... + */ 0)
      }(ExtMinus2Idxs)...
    );
    auto triang_val = _trimap(
      _MDSPAN_FOLD_PLUS_RIGHT(((ExtIdxs == __rank - 2) ? idx : 0), /* + ... + */ 0),
      _MDSPAN_FOLD_PLUS_RIGHT(((ExtIdxs == __rank - 1) ? idx : 0), /* + ... + */ 0)
    );
    return base_val * triang_val;
  }

};

#endif  // 0

}  // end namespace __triangular_layouts_impl

template <class Triangle, class StorageOrder>
class layout_blas_packed;

}}}}  // namespace std::experimental::__p1673_version_0::linalg

#endif  // LINALG_INCLUDE_EXPERIMENTAL_BITS_LAYOUT_TRIANGLE_HPP_
