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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_SOLVE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_SOLVE_HPP_

#include <type_traits>

namespace std { namespace experimental { inline namespace __p1673_version_0 { namespace linalg {

namespace {

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class DiagonalStorage, class ElementType_B, class SizeType_B, ::std::size_t ext_B,
        class Layout_B, class Accessor_B, class ElementType_X, class SizeType_X, ::std::size_t ext_X, class Layout_X,
        class Accessor_X, class BinaryDivideOp>
void trsv_upper_triangular_left_side(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, ext_B>, Layout_B, Accessor_B> B,
        std::experimental::mdspan<ElementType_X, std::experimental::extents<SizeType_X, ext_X>, Layout_X, Accessor_X> X,
        BinaryDivideOp divide) {
    constexpr bool explicit_diagonal = std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
    using size_type = std::common_type_t<SizeType_A, SizeType_B, SizeType_X>;

    const size_type A_num_rows = A.extent(0);
    const size_type A_num_cols = A.extent(1);

    // One advantage of using signed index types is that you can write
    // descending loops with zero-based indices.
    // (AMK 6/8/21) i can't be a nonnegative type because the loop would be infinite
    for (ptrdiff_t i = A_num_rows - 1; i >= 0; --i) {
        // TODO this would be a great opportunity for an implementer to
        // add value, by accumulating in extended precision (or at least
        // in a type with the max precision of X and B).
        using sum_type = decltype(B(i) - A(0, 0) * X(0));
        // using sum_type = typename out_object_t::element_type;
        sum_type t(B(i));
        for (size_type j = i + 1; j < A_num_cols; ++j) {
            t = t - A(i, j) * X(j);
        }
        if constexpr (explicit_diagonal) {
            X(i) = divide(t, A(i, i));
        } else {
            X(i) = t;
        }
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class DiagonalStorage, class ElementType_B, class SizeType_B, ::std::size_t ext_B,
        class Layout_B, class Accessor_B, class ElementType_X, class SizeType_X, ::std::size_t ext_X, class Layout_X,
        class Accessor_X>
void trsv_upper_triangular_left_side(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, ext_B>, Layout_B, Accessor_B> B,
        std::experimental::mdspan<ElementType_X, std::experimental::extents<SizeType_X, ext_X>, Layout_X, Accessor_X>
                X) {
    auto divide = [](const auto& x, const auto& y) { return x / y; };
    trsv_upper_triangular_left_side(A, d, B, X, divide);
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class DiagonalStorage, class ElementType_B, class SizeType_B, ::std::size_t ext_B,
        class Layout_B, class Accessor_B, class ElementType_X, class SizeType_X, ::std::size_t ext_X, class Layout_X,
        class Accessor_X, class BinaryDivideOp>
void trsv_lower_triangular_left_side(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, ext_B>, Layout_B, Accessor_B> B,
        std::experimental::mdspan<ElementType_X, std::experimental::extents<SizeType_X, ext_X>, Layout_X, Accessor_X> X,
        BinaryDivideOp divide) {
    constexpr bool explicit_diagonal = std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
    using size_type = std::common_type_t<SizeType_A, SizeType_B, SizeType_X>;

    const size_type A_num_rows = A.extent(0);
    const size_type A_num_cols = A.extent(1);

    for (size_type i = 0; i < A_num_rows; ++i) {
        // TODO this would be a great opportunity for an implementer to
        // add value, by accumulating in extended precision (or at least
        // in a type with the max precision of X and B).
        using sum_type = decltype(B(i) - A(0, 0) * X(0));
        // using sum_type = typename out_object_t::element_type;
        sum_type t(B(i));
        for (size_type j = 0; j < i; ++j) {
            t = t - A(i, j) * X(j);
        }
        if constexpr (explicit_diagonal) {
            X(i) = divide(t, A(i, i));
        } else {
            X(i) = t;
        }
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class DiagonalStorage, class ElementType_B, class SizeType_B, ::std::size_t ext_B,
        class Layout_B, class Accessor_B, class ElementType_X, class SizeType_X, ::std::size_t ext_X, class Layout_X,
        class Accessor_X>
void trsv_lower_triangular_left_side(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, ext_B>, Layout_B, Accessor_B> B,
        std::experimental::mdspan<ElementType_X, std::experimental::extents<SizeType_X, ext_X>, Layout_X, Accessor_X>
                X) {
    auto divide = [](const auto& x, const auto& y) { return x / y; };
    trsv_lower_triangular_left_side(A, d, B, X, divide);
}

template <class Exec, class A_t, class Tri_t, class D_t, class B_t, class X_t, class = void>
struct is_custom_tri_mat_vec_solve_avail : std::false_type {};

template <class Exec, class A_t, class Tri_t, class D_t, class B_t, class X_t>
struct is_custom_tri_mat_vec_solve_avail<Exec, A_t, Tri_t, D_t, B_t, X_t,
        std::enable_if_t<
                std::is_void_v<decltype(triangular_matrix_vector_solve(std::declval<Exec>(), std::declval<A_t>(),
                        std::declval<Tri_t>(), std::declval<D_t>(), std::declval<B_t>(), std::declval<X_t>()))> &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

}  // end anonymous namespace

// Special case: ExecutionPolicy = inline_exec_t

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class DiagonalStorage, class ElementType_B, class SizeType_B,
        ::std::size_t ext_B, class Layout_B, class Accessor_B, class ElementType_X, class SizeType_X,
        ::std::size_t ext_X, class Layout_X, class Accessor_X, class BinaryDivideOp>
void triangular_matrix_vector_solve(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, ext_B>, Layout_B, Accessor_B> b,
        std::experimental::mdspan<ElementType_X, std::experimental::extents<SizeType_X, ext_X>, Layout_X, Accessor_X> x,
        BinaryDivideOp divide) {
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
        trsv_lower_triangular_left_side(A, d, b, x, divide);
    } else {
        trsv_upper_triangular_left_side(A, d, b, x, divide);
    }
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class DiagonalStorage, class ElementType_B, class SizeType_B,
        ::std::size_t ext_B, class Layout_B, class Accessor_B, class ElementType_X, class SizeType_X,
        ::std::size_t ext_X, class Layout_X, class Accessor_X>
void triangular_matrix_vector_solve(std::experimental::linalg::impl::inline_exec_t&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, ext_B>, Layout_B, Accessor_B> b,
        std::experimental::mdspan<ElementType_X, std::experimental::extents<SizeType_X, ext_X>, Layout_X, Accessor_X>
                x) {
    auto divide = [](const auto& x, const auto& y) { return x / y; };
    triangular_matrix_vector_solve(
            std::forward<std::experimental::linalg::impl::inline_exec_t>(exec), A, t, d, b, x, divide);
}

// Overloads taking an ExecutionPolicy

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class DiagonalStorage,
        class ElementType_B, class SizeType_B, ::std::size_t ext_B, class Layout_B, class Accessor_B,
        class ElementType_X, class SizeType_X, ::std::size_t ext_X, class Layout_X, class Accessor_X,
        class BinaryDivideOp>
void triangular_matrix_vector_solve(ExecutionPolicy&& /* exec */,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, ext_B>, Layout_B, Accessor_B> b,
        std::experimental::mdspan<ElementType_X, std::experimental::extents<SizeType_X, ext_X>, Layout_X, Accessor_X> x,
        BinaryDivideOp divide) {
    // FIXME (mfh 2022/06/13) We don't yet have a parallel version
    // that takes a generic divide operator.
    triangular_matrix_vector_solve(std::experimental::linalg::impl::inline_exec_t {}, A, t, d, b, x, divide);
}

template <class ExecutionPolicy, class ElementType_A, class SizeType_A, ::std::size_t numRows_A,
        ::std::size_t numCols_A, class Layout_A, class Accessor_A, class Triangle, class DiagonalStorage,
        class ElementType_B, class SizeType_B, ::std::size_t ext_B, class Layout_B, class Accessor_B,
        class ElementType_X, class SizeType_X, ::std::size_t ext_X, class Layout_X, class Accessor_X>
void triangular_matrix_vector_solve(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, ext_B>, Layout_B, Accessor_B> b,
        std::experimental::mdspan<ElementType_X, std::experimental::extents<SizeType_X, ext_X>, Layout_X, Accessor_X>
                x) {
    constexpr bool use_custom = is_custom_tri_mat_vec_solve_avail<decltype(execpolicy_mapper(exec)), decltype(A),
            decltype(t), decltype(d), decltype(b), decltype(x)>::value;

    if constexpr (use_custom) {
        triangular_matrix_vector_solve(execpolicy_mapper(exec), A, t, d, b, x);
    } else {
        triangular_matrix_vector_solve(std::experimental::linalg::impl::inline_exec_t(), A, t, d, b, x);
    }
}

// Overloads not taking an ExecutionPolicy

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class DiagonalStorage, class ElementType_B, class SizeType_B,
        ::std::size_t ext_B, class Layout_B, class Accessor_B, class ElementType_X, class SizeType_X,
        ::std::size_t ext_X, class Layout_X, class Accessor_X, class BinaryDivideOp>
void triangular_matrix_vector_solve(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, ext_B>, Layout_B, Accessor_B> b,
        std::experimental::mdspan<ElementType_X, std::experimental::extents<SizeType_X, ext_X>, Layout_X, Accessor_X> x,
        BinaryDivideOp divide) {
    triangular_matrix_vector_solve(std::experimental::linalg::impl::default_exec_t(), A, t, d, b, x, divide);
}

template <class ElementType_A, class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A,
        class Accessor_A, class Triangle, class DiagonalStorage, class ElementType_B, class SizeType_B,
        ::std::size_t ext_B, class Layout_B, class Accessor_B, class ElementType_X, class SizeType_X,
        ::std::size_t ext_X, class Layout_X, class Accessor_X>
void triangular_matrix_vector_solve(
        std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A,
                Accessor_A>
                A,
        Triangle t, DiagonalStorage d,
        std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, ext_B>, Layout_B, Accessor_B> b,
        std::experimental::mdspan<ElementType_X, std::experimental::extents<SizeType_X, ext_X>, Layout_X, Accessor_X>
                x) {
    triangular_matrix_vector_solve(std::experimental::linalg::impl::default_exec_t(), A, t, d, b, x);
}

}}}}  // namespace std::experimental::__p1673_version_0::linalg

#endif  // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_SOLVE_HPP_
