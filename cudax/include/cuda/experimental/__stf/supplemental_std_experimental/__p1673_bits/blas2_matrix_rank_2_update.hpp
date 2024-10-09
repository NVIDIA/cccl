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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_RANK_2_UPDATE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_RANK_2_UPDATE_HPP_

namespace std { namespace experimental { inline namespace __p1673_version_0 { namespace linalg {

namespace {

template <class Exec, class x_t, class y_t, class A_t, class Tr_t, class = void>
struct is_custom_symmetric_matrix_rank_2_update_avail : std::false_type {};

template <class Exec, class x_t, class y_t, class A_t, class Tr_t>
struct is_custom_symmetric_matrix_rank_2_update_avail<Exec, x_t, y_t, A_t, Tr_t,
        std::enable_if_t<
                std::is_void_v<decltype(symmetric_matrix_rank_2_update(std::declval<Exec>(), std::declval<x_t>(),
                        std::declval<y_t>(), std::declval<A_t>(), std::declval<Tr_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class Exec, class x_t, class y_t, class A_t, class Tr_t, class = void>
struct is_custom_hermitian_matrix_rank_2_update_avail : std::false_type {};

template <class Exec, class x_t, class y_t, class A_t, class Tr_t>
struct is_custom_hermitian_matrix_rank_2_update_avail<Exec, x_t, y_t, A_t, Tr_t,
        std::enable_if_t<
                std::is_void_v<decltype(hermitian_matrix_rank_2_update(std::declval<Exec>(), std::declval<x_t>(),
                        std::declval<y_t>(), std::declval<A_t>(), std::declval<Tr_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

}  // end anonymous namespace

// Rank-2 update of a symmetric matrix

template <class ElementType_x, class SizeType_x, ::std::size_t ext_x, class Layout_x, class Accessor_x,
        class ElementType_y, class SizeType_y, ::std::size_t ext_y, class Layout_y, class Accessor_y,
        class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle>
void symmetric_matrix_rank_2_update(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */) {
    using size_type = std::common_type_t<SizeType_x, SizeType_y, SizeType_A>;
    constexpr bool lower_tri = std::is_same_v<Triangle, lower_triangle_t>;
    for (size_type j = 0; j < A.extent(1); ++j) {
        const size_type i_lower = lower_tri ? j : size_type(0);
        const size_type i_upper = lower_tri ? A.extent(0) : j + 1;

        for (size_type i = i_lower; i < i_upper; ++i) {
            A(i, j) += x(i) * y(j) + y(i) * x(j);
        }
    }
}

template <class ExecutionPolicy, class ElementType_x, class SizeType_x, ::std::size_t ext_x, class Layout_x,
        class Accessor_x, class ElementType_y, class SizeType_y, ::std::size_t ext_y, class Layout_y, class Accessor_y,
        class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle>
void symmetric_matrix_rank_2_update(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t) {
    constexpr bool use_custom = is_custom_symmetric_matrix_rank_2_update_avail<decltype(execpolicy_mapper(exec)),
            decltype(x), decltype(y), decltype(A), Triangle>::value;

    if constexpr (use_custom) {
        symmetric_matrix_rank_2_update(execpolicy_mapper(exec), x, y, A, t);
    } else {
        symmetric_matrix_rank_2_update(std::experimental::linalg::impl::inline_exec_t(), x, y, A, t);
    }
}

template <class ElementType_x, class SizeType_x, ::std::size_t ext_x, class Layout_x, class Accessor_x,
        class ElementType_y, class SizeType_y, ::std::size_t ext_y, class Layout_y, class Accessor_y,
        class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle>
void symmetric_matrix_rank_2_update(
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t) {
    symmetric_matrix_rank_2_update(std::experimental::linalg::impl::default_exec_t(), x, y, A, t);
}

// Rank-2 update of a Hermitian matrix

template <class ElementType_x, class SizeType_x, ::std::size_t ext_x, class Layout_x, class Accessor_x,
        class ElementType_y, class SizeType_y, ::std::size_t ext_y, class Layout_y, class Accessor_y,
        class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle>
void hermitian_matrix_rank_2_update(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle /* t */) {
    using size_type = std::common_type_t<SizeType_x, SizeType_y, SizeType_A>;

    constexpr bool lower_tri = std::is_same_v<Triangle, lower_triangle_t>;
    for (size_type j = 0; j < A.extent(1); ++j) {
        const size_type i_lower = lower_tri ? j : size_type(0);
        const size_type i_upper = lower_tri ? A.extent(0) : j + 1;

        for (size_type i = i_lower; i < i_upper; ++i) {
            A(i, j) += x(i) * impl::conj_if_needed(y(j)) + y(i) * impl::conj_if_needed(x(j));
        }
    }
}

template <class ExecutionPolicy, class ElementType_x, class SizeType_x, ::std::size_t ext_x, class Layout_x,
        class Accessor_x, class ElementType_y, class SizeType_y, ::std::size_t ext_y, class Layout_y, class Accessor_y,
        class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle>
void hermitian_matrix_rank_2_update(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t) {
    constexpr bool use_custom = is_custom_hermitian_matrix_rank_2_update_avail<decltype(execpolicy_mapper(exec)),
            decltype(x), decltype(y), decltype(A), Triangle>::value;

    if constexpr (use_custom) {
        hermitian_matrix_rank_2_update(execpolicy_mapper(exec), x, y, A, t);
    } else {
        hermitian_matrix_rank_2_update(std::experimental::linalg::impl::inline_exec_t(), x, y, A, t);
    }
}

template <class ElementType_x, class SizeType_x, ::std::size_t ext_x, class Layout_x, class Accessor_x,
        class ElementType_y, class SizeType_y, ::std::size_t ext_y, class Layout_y, class Accessor_y,
        class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle>
void hermitian_matrix_rank_2_update(
        std::experimental::mdspan<ElementType_x, std::experimental::extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
        std::experimental::mdspan<ElementType_y, std::experimental::extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t) {
    hermitian_matrix_rank_2_update(std::experimental::linalg::impl::default_exec_t(), x, y, A, t);
}

}}}}  // namespace std::experimental::__p1673_version_0::linalg

#endif  // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_RANK_2_UPDATE_HPP_
