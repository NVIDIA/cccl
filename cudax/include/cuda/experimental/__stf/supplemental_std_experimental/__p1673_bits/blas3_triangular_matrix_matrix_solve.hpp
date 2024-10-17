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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_HPP_

namespace std
{
namespace experimental
{
inline namespace __p1673_version_0
{
namespace linalg
{

namespace
{

template <class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class DiagonalStorage,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void trsm_upper_triangular_left_side(
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  DiagonalStorage d,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  constexpr bool explicit_diagonal = std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type                  = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_X>;

  const size_type A_num_rows = A.extent(0);
  const size_type B_num_cols = B.extent(1);

  for (size_type k = 0; k < B_num_cols; ++k)
  {
    // One advantage of using signed index types is that you can write
    // descending loops with zero-based indices.
    // (AMK 6/8/21) i can't be a nonnegative type because the loop would be infinite
    for (ptrdiff_t i = A_num_rows - 1; i >= 0; --i)
    {
      // TODO this would be a great opportunity for an implementer to
      // add value, by accumulating in extended precision (or at least
      // in a type with the max precision of X and B).
      using sum_type = decltype(B(i, k) - A(0, 0) * X(0, 0));
      // using sum_type = typename out_object_t::element_type;
      sum_type t(B(i, k));
      for (size_type j = i + 1; j < A_num_rows; ++j)
      {
        t = t - A(i, j) * X(j, k);
      }
      if constexpr (explicit_diagonal)
      {
        X(i, k) = t / A(i, i);
      }
      else
      {
        X(i, k) = t;
      }
    }
  }
}

template <class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class DiagonalStorage,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void trsm_lower_triangular_left_side(
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  DiagonalStorage d,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  constexpr bool explicit_diagonal = std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type                  = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_X>;

  const size_type A_num_rows = A.extent(0);
  const size_type B_num_cols = B.extent(1);

  for (size_type k = 0; k < B_num_cols; ++k)
  {
    for (size_type i = 0; i < A_num_rows; ++i)
    {
      // TODO this would be a great opportunity for an implementer to
      // add value, by accumulating in extended precision (or at least
      // in a type with the max precision of X and B).
      ElementType_X t(B(i, k));
      for (size_type j = 0; j < i; ++j)
      {
        t = t - A(i, j) * X(j, k);
      }
      if constexpr (explicit_diagonal)
      {
        X(i, k) = t / A(i, i);
      }
      else
      {
        X(i, k) = t;
      }
    }
  }
}

template <class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class DiagonalStorage,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void trsm_upper_triangular_right_side(
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  DiagonalStorage d,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  constexpr bool explicit_diagonal = std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type                  = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_X>;

  const size_type B_num_rows = B.extent(0);
  const size_type A_num_cols = A.extent(1);

  for (size_type i = 0; i < B_num_rows; ++i)
  {
    for (size_type j = 0; j < A_num_cols; ++j)
    {
      using sum_type = decltype(B(i, j) - A(0, 0) * X(0, 0));
      sum_type t(B(i, j));
      for (size_type k = 0; k < j; ++k)
      {
        t = t - X(i, k) * A(k, j);
      }
      if constexpr (explicit_diagonal)
      {
        X(i, j) = t / A(j, j);
      }
      else
      {
        X(i, j) = t;
      }
    }
  }
}

template <class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class DiagonalStorage,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void trsm_lower_triangular_right_side(
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  DiagonalStorage d,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  constexpr bool explicit_diagonal = std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type                  = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_X>;

  const size_type B_num_rows = B.extent(0);
  const size_type A_num_rows = A.extent(0);
  const size_type A_num_cols = A.extent(1);

  for (size_type i = 0; i < B_num_rows; ++i)
  {
    for (ptrdiff_t j = A_num_cols - 1; j >= 0; --j)
    {
      using sum_type = decltype(B(i, j) - A(0, 0) * X(0, 0));
      sum_type t(B(i, j));
      for (size_type k = j + 1; k < A_num_rows; ++k)
      {
        t = t - X(i, k) * A(k, j);
      }
      if constexpr (explicit_diagonal)
      {
        X(i, j) = t / A(j, j);
      }
      else
      {
        X(i, j) = t;
      }
    }
  }
}

template <class Exec, class A_t, class Tri_t, class D_t, class B_t, class X_t, class = void>
struct is_custom_tri_matrix_matrix_left_solve_avail : std::false_type
{};

template <class Exec, class A_t, class Tri_t, class D_t, class B_t, class X_t>
struct is_custom_tri_matrix_matrix_left_solve_avail<
  Exec,
  A_t,
  Tri_t,
  D_t,
  B_t,
  X_t,
  std::enable_if_t<std::is_void_v<decltype(triangular_matrix_matrix_left_solve(std::declval<Exec>(),
                                                                               std::declval<A_t>(),
                                                                               std::declval<Tri_t>(),
                                                                               std::declval<D_t>(),
                                                                               std::declval<B_t>(),
                                                                               std::declval<X_t>()))>
                   && !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type
{};

template <class Exec, class A_t, class Tri_t, class D_t, class B_t, class X_t, class = void>
struct is_custom_tri_matrix_matrix_right_solve_avail : std::false_type
{};

template <class Exec, class A_t, class Tri_t, class D_t, class B_t, class X_t>
struct is_custom_tri_matrix_matrix_right_solve_avail<
  Exec,
  A_t,
  Tri_t,
  D_t,
  B_t,
  X_t,
  std::enable_if_t<std::is_void_v<decltype(triangular_matrix_matrix_right_solve(std::declval<Exec>(),
                                                                                std::declval<A_t>(),
                                                                                std::declval<Tri_t>(),
                                                                                std::declval<D_t>(),
                                                                                std::declval<B_t>(),
                                                                                std::declval<X_t>()))>
                   && !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type
{};

template <class Exec, class A_t, class Tri_t, class D_t, class Side_t, class B_t, class X_t, class = void>
struct is_custom_tri_matrix_matrix_solve_avail : std::false_type
{};

template <class Exec, class A_t, class Tri_t, class D_t, class Side_t, class B_t, class X_t>
struct is_custom_tri_matrix_matrix_solve_avail<
  Exec,
  A_t,
  Tri_t,
  D_t,
  Side_t,
  B_t,
  X_t,
  std::enable_if_t<std::is_void_v<decltype(triangular_matrix_matrix_right_solve(std::declval<Exec>(),
                                                                                std::declval<A_t>(),
                                                                                std::declval<Tri_t>(),
                                                                                std::declval<D_t>(),
                                                                                std::declval<Side_t>(),
                                                                                std::declval<B_t>(),
                                                                                std::declval<X_t>()))>
                   && !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type
{};

} // end anonymous namespace

// triangular_matrix_matrix_left_solve

template <class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class Triangle,
          class DiagonalStorage,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void triangular_matrix_matrix_left_solve(
  std::experimental::linalg::impl::inline_exec_t&& /* exec */,
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /*t*/,
  DiagonalStorage d,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  if (std::is_same_v<Triangle, lower_triangle_t>)
  {
    trsm_lower_triangular_left_side(A, d, B, X);
  }
  else
  {
    trsm_upper_triangular_left_side(A, d, B, X);
  }
}

template <class ExecutionPolicy,
          class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class Triangle,
          class DiagonalStorage,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void triangular_matrix_matrix_left_solve(
  ExecutionPolicy&& exec,
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  constexpr bool use_custom = is_custom_tri_matrix_matrix_left_solve_avail<
    decltype(execpolicy_mapper(exec)),
    decltype(A),
    Triangle,
    DiagonalStorage,
    decltype(B),
    decltype(X)>::value;

  if constexpr (use_custom)
  {
    triangular_matrix_matrix_left_solve(execpolicy_mapper(exec), A, t, d, B, X);
  }
  else
  {
    triangular_matrix_matrix_left_solve(std::experimental::linalg::impl::inline_exec_t(), A, t, d, B, X);
  }
}

template <class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class Triangle,
          class DiagonalStorage,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void triangular_matrix_matrix_left_solve(
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  triangular_matrix_matrix_left_solve(std::experimental::linalg::impl::default_exec_t(), A, t, d, B, X);
}

// triangular_matrix_matrix_right_solve

template <class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class Triangle,
          class DiagonalStorage,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void triangular_matrix_matrix_right_solve(
  std::experimental::linalg::impl::inline_exec_t&& /* exec */,
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /*t*/,
  DiagonalStorage d,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  if (std::is_same_v<Triangle, lower_triangle_t>)
  {
    trsm_lower_triangular_right_side(A, d, B, X);
  }
  else
  {
    trsm_upper_triangular_right_side(A, d, B, X);
  }
}

template <class ExecutionPolicy,
          class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class Triangle,
          class DiagonalStorage,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void triangular_matrix_matrix_right_solve(
  ExecutionPolicy&& exec,
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  constexpr bool use_custom = is_custom_tri_matrix_matrix_right_solve_avail<
    decltype(execpolicy_mapper(exec)),
    decltype(A),
    Triangle,
    DiagonalStorage,
    decltype(B),
    decltype(X)>::value;

  if constexpr (use_custom)
  {
    triangular_matrix_matrix_right_solve(execpolicy_mapper(exec), A, t, d, B, X);
  }
  else
  {
    triangular_matrix_matrix_right_solve(std::experimental::linalg::impl::inline_exec_t(), A, t, d, B, X);
  }
}

template <class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class Triangle,
          class DiagonalStorage,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void triangular_matrix_matrix_right_solve(
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  triangular_matrix_matrix_right_solve(std::experimental::linalg::impl::default_exec_t(), A, t, d, B, X);
}

// triangular_matrix_matrix_solve

template <class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class Triangle,
          class DiagonalStorage,
          class Side,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void triangular_matrix_matrix_solve(
  std::experimental::linalg::impl::inline_exec_t&& /* exec */,
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /*t*/,
  DiagonalStorage d,
  Side /*s*/,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  if (std::is_same_v<Side, left_side_t>)
  {
    triangular_matrix_matrix_left_solve(A, d, B, X);
  }
  else
  {
    triangular_matrix_matrix_right_solve(A, d, B, X);
  }
}

template <class ExecutionPolicy,
          class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class Triangle,
          class DiagonalStorage,
          class Side,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void triangular_matrix_matrix_solve(
  ExecutionPolicy&& exec,
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  Side s,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  constexpr bool use_custom = is_custom_tri_matrix_matrix_solve_avail<
    decltype(execpolicy_mapper(exec)),
    decltype(A),
    Triangle,
    DiagonalStorage,
    Side,
    decltype(B),
    decltype(X)>::value;

  if constexpr (use_custom)
  {
    triangular_matrix_matrix_solve(execpolicy_mapper(exec), A, t, d, s, B, X);
  }
  else
  {
    triangular_matrix_matrix_solve(std::experimental::linalg::impl::inline_exec_t(), A, t, d, s, B, X);
  }
}

template <class ElementType_A,
          class SizeType_A,
          ::std::size_t numRows_A,
          ::std::size_t numCols_A,
          class Layout_A,
          class Accessor_A,
          class Triangle,
          class DiagonalStorage,
          class Side,
          class ElementType_B,
          class SizeType_B,
          ::std::size_t numRows_B,
          ::std::size_t numCols_B,
          class Layout_B,
          class Accessor_B,
          class ElementType_X,
          class SizeType_X,
          ::std::size_t numRows_X,
          ::std::size_t numCols_X,
          class Layout_X,
          class Accessor_X>
void triangular_matrix_matrix_solve(
  std::experimental::
    mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  Side s,
  std::experimental::
    mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::
    mdspan<ElementType_X, std::experimental::extents<SizeType_X, numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  triangular_matrix_matrix_solve(std::experimental::linalg::impl::default_exec_t(), A, t, d, s, B, X);
}

} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std

#endif // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_HPP_
