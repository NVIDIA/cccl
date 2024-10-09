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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_ADD_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_ADD_HPP_

namespace std { namespace experimental { inline namespace __p1673_version_0 { namespace linalg {

namespace {

template <class ElementType_x, class SizeType_x, ::std::size_t ext_x, class Layout_x, class Accessor_x,
        class ElementType_y, class SizeType_y, ::std::size_t ext_y, class Layout_y, class Accessor_y,
        class ElementType_z, class SizeType_z, ::std::size_t ext_z, class Layout_z, class Accessor_z>
void add_rank_1(
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
        std::experimental::mdspan<ElementType_z, std::experimental::extents<SizeType_z, ext_z>, Layout_z, Accessor_z>
                z) {
    static_assert(x.static_extent(0) == dynamic_extent || z.static_extent(0) == dynamic_extent ||
                  x.static_extent(0) == z.static_extent(0));
    static_assert(y.static_extent(0) == dynamic_extent || z.static_extent(0) == dynamic_extent ||
                  y.static_extent(0) == z.static_extent(0));
    static_assert(x.static_extent(0) == dynamic_extent || y.static_extent(0) == dynamic_extent ||
                  x.static_extent(0) == y.static_extent(0));

    using size_type = std::common_type_t<SizeType_x, SizeType_y, SizeType_z>;
    for (size_type i = 0; i < z.extent(0); ++i) {
        z(i) = x(i) + y(i);
    }
}

template <class ElementType_x, class SizeType_x, ::std::size_t numRows_x, ::std::size_t numCols_x, class Layout_x,
        class Accessor_x, class ElementType_y, class SizeType_y, ::std::size_t numRows_y, ::std::size_t numCols_y,
        class Layout_y, class Accessor_y, class ElementType_z, class SizeType_z, ::std::size_t numRows_z,
        ::std::size_t numCols_z, class Layout_z, class Accessor_z>
void add_rank_2(std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, numRows_x, numCols_x>,
                        Layout_x, Accessor_x>
                        x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, numRows_y, numCols_y>, Layout_y,
                Accessor_y>
                y,
        std::experimental::mdspan<ElementType_z, std::experimental::extents<SizeType_z, numRows_z, numCols_z>, Layout_z,
                Accessor_z>
                z) {
    static_assert(x.static_extent(0) == dynamic_extent || z.static_extent(0) == dynamic_extent ||
                  x.static_extent(0) == z.static_extent(0));
    static_assert(y.static_extent(0) == dynamic_extent || z.static_extent(0) == dynamic_extent ||
                  y.static_extent(0) == z.static_extent(0));
    static_assert(x.static_extent(0) == dynamic_extent || y.static_extent(0) == dynamic_extent ||
                  x.static_extent(0) == y.static_extent(0));

    static_assert(x.static_extent(1) == dynamic_extent || z.static_extent(1) == dynamic_extent ||
                  x.static_extent(1) == z.static_extent(1));
    static_assert(y.static_extent(1) == dynamic_extent || z.static_extent(1) == dynamic_extent ||
                  y.static_extent(1) == z.static_extent(1));
    static_assert(x.static_extent(1) == dynamic_extent || y.static_extent(1) == dynamic_extent ||
                  x.static_extent(1) == y.static_extent(1));

    using size_type = std::common_type_t<SizeType_x, SizeType_y, SizeType_z>;
    for (size_type j = 0; j < x.extent(1); ++j) {
        for (size_type i = 0; i < x.extent(0); ++i) {
            z(i, j) = x(i, j) + y(i, j);
        }
    }
}

template <class Exec, class x_t, class y_t, class z_t, class = void>
struct is_custom_add_avail : std::false_type {};

template <class Exec, class x_t, class y_t, class z_t>
struct is_custom_add_avail<Exec, x_t, y_t, z_t,
        std::enable_if_t<std::is_void_v<decltype(add(std::declval<Exec>(), std::declval<x_t>(), std::declval<y_t>(),
                                 std::declval<z_t>()))> &&
                         !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

}  // end anonymous namespace

MDSPAN_TEMPLATE_REQUIRES(class ElementType_x, class SizeType_x, ::std::size_t... ext_x, class Layout_x,
        class Accessor_x, class ElementType_y, class SizeType_y, ::std::size_t... ext_y, class Layout_y,
        class Accessor_y, class ElementType_z, class SizeType_z, ::std::size_t... ext_z, class Layout_z,
        class Accessor_z,
        /* requires */ (sizeof...(ext_x) == sizeof...(ext_y) && sizeof...(ext_x) == sizeof...(ext_z)))
void add(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x...>, Layout_x, Accessor_x>
                x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y...>, Layout_y, Accessor_y>
                y,
        std::experimental::mdspan<ElementType_z, std::experimental::extents<SizeType_z, ext_z...>, Layout_z, Accessor_z>
                z) {
    // this static assert is only here because for
    // the default case we support rank-1 and rank2.
    static_assert(z.rank() <= 2);

    if constexpr (z.rank() == 1) {
        add_rank_1(x, y, z);
    } else if constexpr (z.rank() == 2) {
        add_rank_2(x, y, z);
    }
}

MDSPAN_TEMPLATE_REQUIRES(class ExecutionPolicy, class ElementType_x, class SizeType_x, ::std::size_t... ext_x,
        class Layout_x, class Accessor_x, class ElementType_y, class SizeType_y, ::std::size_t... ext_y, class Layout_y,
        class Accessor_y, class ElementType_z, class SizeType_z, ::std::size_t... ext_z, class Layout_z,
        class Accessor_z,
        /* requires */ (sizeof...(ext_x) == sizeof...(ext_y) && sizeof...(ext_x) == sizeof...(ext_z)))
void add(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x...>, Layout_x, Accessor_x>
                x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y...>, Layout_y, Accessor_y>
                y,
        std::experimental::mdspan<ElementType_z, std::experimental::extents<SizeType_z, ext_z...>, Layout_z, Accessor_z>
                z) {
    constexpr bool use_custom =
            is_custom_add_avail<decltype(execpolicy_mapper(exec)), decltype(x), decltype(y), decltype(z)>::value;

    if constexpr (use_custom) {
        // for the customization point, it is up to impl to check requirements
        add(execpolicy_mapper(exec), x, y, z);
    } else {
        add(std::experimental::linalg::impl::inline_exec_t(), x, y, z);
    }
}

MDSPAN_TEMPLATE_REQUIRES(class ElementType_x, class SizeType_x, ::std::size_t... ext_x, class Layout_x,
        class Accessor_x, class ElementType_y, class SizeType_y, ::std::size_t... ext_y, class Layout_y,
        class Accessor_y, class ElementType_z, class SizeType_z, ::std::size_t... ext_z, class Layout_z,
        class Accessor_z,
        /* requires */ (sizeof...(ext_x) == sizeof...(ext_y) && sizeof...(ext_x) == sizeof...(ext_z)))
void add(
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x...>, Layout_x, Accessor_x>
                x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y...>, Layout_y, Accessor_y>
                y,
        std::experimental::mdspan<ElementType_z, std::experimental::extents<SizeType_z, ext_z...>, Layout_z, Accessor_z>
                z) {
    add(std::experimental::linalg::impl::default_exec_t(), x, y, z);
}

}}}}  // namespace std::experimental::__p1673_version_0::linalg

#endif  // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_ADD_HPP_
