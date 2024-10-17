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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_MATRIX_FROB_NORM_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_MATRIX_FROB_NORM_HPP_

#include <cmath>
#include <cstdlib>

namespace std
{
namespace experimental
{
inline namespace __p1673_version_0
{
namespace linalg
{

// begin anonymous namespace
namespace
{

template <class Exec, class A_t, class Scalar, class = void>
struct is_custom_matrix_frob_norm_avail : std::false_type
{};

template <class Exec, class A_t, class Scalar>
struct is_custom_matrix_frob_norm_avail<
  Exec,
  A_t,
  Scalar,
  std::enable_if_t<
    std::is_same<decltype(matrix_frob_norm(std::declval<Exec>(), std::declval<A_t>(), std::declval<Scalar>())), Scalar>::value
    && !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type
{};
} // end anonymous namespace

template <class ElementType,
          class SizeType,
          ::std::size_t numRows,
          ::std::size_t numCols,
          class Layout,
          class Accessor,
          class Scalar>
Scalar matrix_frob_norm(
  std::experimental::linalg::impl::inline_exec_t&& /* exec */,
  std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, numRows, numCols>, Layout, Accessor> A,
  Scalar init)
{
  using std::abs;
  using std::sqrt;
  using size_type = SizeType;

  // Handle special cases.
  auto result = init;
  if (A.extent(0) == 0 || A.extent(1) == 0)
  {
    return result;
  }
  else if (A.extent(0) == size_type(1) && A.extent(1) == size_type(1))
  {
    result += abs(A(0, 0));
    return result;
  }

  // Rescaling avoids unwarranted overflow or underflow.
  Scalar scale = 0.0;
  Scalar ssq   = 1.0;
  for (size_type i = 0; i < A.extent(0); ++i)
  {
    for (size_type j = 0; j < A.extent(1); ++j)
    {
      const auto absaij = abs(A(i, j));
      if (absaij != 0.0)
      {
        const auto quotient = scale / absaij;
        if (scale < absaij)
        {
          ssq   = Scalar(1.0) + ssq * quotient * quotient;
          scale = absaij;
        }
        else
        {
          ssq = ssq + quotient * quotient;
        }
      }
    }
  }
  result += scale * sqrt(ssq);
  return result;
}

template <class ExecutionPolicy,
          class ElementType,
          class SizeType,
          ::std::size_t numRows,
          ::std::size_t numCols,
          class Layout,
          class Accessor,
          class Scalar>
Scalar matrix_frob_norm(
  ExecutionPolicy&& exec,
  std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, numRows, numCols>, Layout, Accessor> A,
  Scalar init)
{
  constexpr bool use_custom =
    is_custom_matrix_frob_norm_avail<decltype(execpolicy_mapper(exec)), decltype(A), Scalar>::value;

  if constexpr (use_custom)
  {
    return matrix_frob_norm(execpolicy_mapper(exec), A, init);
  }
  else
  {
    return matrix_frob_norm(std::experimental::linalg::impl::inline_exec_t(), A, init);
  }
}

template <class ElementType,
          class SizeType,
          ::std::size_t numRows,
          ::std::size_t numCols,
          class Layout,
          class Accessor,
          class Scalar>
Scalar matrix_frob_norm(
  std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, numRows, numCols>, Layout, Accessor> A,
  Scalar init)
{
  return matrix_frob_norm(std::experimental::linalg::impl::default_exec_t(), A, init);
}

namespace matrix_frob_norm_detail
{

// The point of this is to do correct ADL for abs,
// without exposing "using std::abs" in the outer namespace.
using std::abs;
template <class ElementType, class SizeType, ::std::size_t numRows, ::std::size_t numCols, class Layout, class Accessor>
auto matrix_frob_norm_return_type_deducer(
  std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, numRows, numCols>, Layout, Accessor> A)
  -> decltype(abs(A(0, 0)) * abs(A(0, 0)));

} // namespace matrix_frob_norm_detail

template <class ElementType, class SizeType, ::std::size_t numRows, ::std::size_t numCols, class Layout, class Accessor>
auto matrix_frob_norm(
  std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, numRows, numCols>, Layout, Accessor> A)
  -> decltype(matrix_frob_norm_detail::matrix_frob_norm_return_type_deducer(A))
{
  using return_t = decltype(matrix_frob_norm_detail::matrix_frob_norm_return_type_deducer(A));
  return matrix_frob_norm(A, return_t{});
}

template <class ExecutionPolicy,
          class ElementType,
          class SizeType,
          ::std::size_t numRows,
          ::std::size_t numCols,
          class Layout,
          class Accessor>
auto matrix_frob_norm(
  ExecutionPolicy&& exec,
  std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, numRows, numCols>, Layout, Accessor> A)
  -> decltype(matrix_frob_norm_detail::matrix_frob_norm_return_type_deducer(A))
{
  using return_t = decltype(matrix_frob_norm_detail::matrix_frob_norm_return_type_deducer(A));
  return matrix_frob_norm(exec, A, return_t{});
}

} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std

#endif // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_MATRIX_FROB_NORM_HPP_
