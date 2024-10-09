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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_SWAP_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_SWAP_HPP_

#include <utility>

namespace std { namespace experimental { inline namespace __p1673_version_0 { namespace linalg {

namespace {

template <class ElementType_x, class SizeType_x, ::std::size_t ext_x, class Layout_x, class Accessor_x,
        class ElementType_y, class SizeType_y, ::std::size_t ext_y, class Layout_y, class Accessor_y>
void swap_rank_1(
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y>, Layout_y, Accessor_y>
                y) {
    static_assert(x.static_extent(0) == dynamic_extent || y.static_extent(0) == dynamic_extent ||
                  x.static_extent(0) == y.static_extent(0));

    using std::swap;
    using size_type = std::common_type_t<SizeType_x, SizeType_y>;

    for (size_type i = 0; i < y.extent(0); ++i) {
        swap(x(i), y(i));
    }
}

template <class ElementType_x, class SizeType_x, ::std::size_t numRows_x, ::std::size_t numCols_x, class Layout_x,
        class Accessor_x, class ElementType_y, class SizeType_y, ::std::size_t numRows_y, ::std::size_t numCols_y,
        class Layout_y, class Accessor_y>
void swap_rank_2(std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, numRows_x, numCols_x>,
                         Layout_x, Accessor_x>
                         x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, numRows_y, numCols_y>, Layout_y,
                Accessor_y>
                y) {
    static_assert(x.static_extent(0) == dynamic_extent || y.static_extent(0) == dynamic_extent ||
                  x.static_extent(0) == y.static_extent(0));
    static_assert(x.static_extent(1) == dynamic_extent || y.static_extent(1) == dynamic_extent ||
                  x.static_extent(1) == y.static_extent(1));

    using std::swap;
    using size_type = ::std::common_type_t<SizeType_x, SizeType_y>;

    for (size_type j = 0; j < y.extent(1); ++j) {
        for (size_type i = 0; i < y.extent(0); ++i) {
            swap(x(i, j), y(i, j));
        }
    }
}

template <class Exec, class x_t, class y_t, class = void>
struct is_custom_vector_swap_elements_avail : std::false_type {};

template <class Exec, class x_t, class y_t>
struct is_custom_vector_swap_elements_avail<Exec, x_t, y_t,
        std::enable_if_t<std::is_void_v<decltype(swap_elements(
                                 std::declval<Exec>(), std::declval<x_t>(), std::declval<y_t>()))> &&
                         !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

}  // end anonymous namespace

MDSPAN_TEMPLATE_REQUIRES(class ElementType_x, class SizeType_x, ::std::size_t... ext_x, class Layout_x,
        class Accessor_x, class ElementType_y, class SizeType_y, ::std::size_t... ext_y, class Layout_y,
        class Accessor_y,
        /* requires */ (sizeof...(ext_x) == sizeof...(ext_y)))
void swap_elements(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x...>, Layout_x, Accessor_x>
                x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y...>, Layout_y, Accessor_y>
                y) {
    static_assert(x.rank() <= 2);

    if constexpr (x.rank() == 1 && y.rank() == 1) {
        swap_rank_1(x, y);
    } else if constexpr (x.rank() == 2 && y.rank() == 2) {
        swap_rank_2(x, y);
    }
}

MDSPAN_TEMPLATE_REQUIRES(class ExecutionPolicy, class ElementType_x, class SizeType_x, ::std::size_t... ext_x,
        class Layout_x, class Accessor_x, class ElementType_y, class SizeType_y, ::std::size_t... ext_y, class Layout_y,
        class Accessor_y,
        /* requires */ (sizeof...(ext_x) == sizeof...(ext_y)))
void swap_elements(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x...>, Layout_x, Accessor_x>
                x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y...>, Layout_y, Accessor_y>
                y) {
    constexpr bool use_custom =
            is_custom_vector_swap_elements_avail<decltype(execpolicy_mapper(exec)), decltype(x), decltype(y)>::value;

    if constexpr (use_custom) {
        return swap_elements(execpolicy_mapper(exec), x, y);
    } else {
        return swap_elements(std::experimental::linalg::impl::inline_exec_t(), x, y);
    }
}

MDSPAN_TEMPLATE_REQUIRES(class ElementType_x, class SizeType_x, ::std::size_t... ext_x, class Layout_x,
        class Accessor_x, class ElementType_y, class SizeType_y, ::std::size_t... ext_y, class Layout_y,
        class Accessor_y,
        /* requires */ (sizeof...(ext_x) == sizeof...(ext_y)))
void swap_elements(
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x...>, Layout_x, Accessor_x>
                x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y...>, Layout_y, Accessor_y>
                y) {
    swap_elements(std::experimental::linalg::impl::default_exec_t(), x, y);
}

}}}}  // namespace std::experimental::__p1673_version_0::linalg

#endif  // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_SWAP_HPP_
