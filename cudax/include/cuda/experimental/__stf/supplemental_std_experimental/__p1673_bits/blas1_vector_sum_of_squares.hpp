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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_SUM_OF_SQUARES_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_SUM_OF_SQUARES_HPP_

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

// Scaled sum of squares of a vector's elements
template <class Scalar>
struct sum_of_squares_result
{
  Scalar scaling_factor;
  Scalar scaled_sum_of_squares;
};

namespace
{
template <class Exec, class x_t, class Scalar, class = void>
struct is_custom_vector_sum_of_squares_avail : std::false_type
{};

template <class Exec, class x_t, class Scalar>
struct is_custom_vector_sum_of_squares_avail<
  Exec,
  x_t,
  Scalar,
  std::enable_if_t<
    std::is_same<decltype(vector_sum_of_squares(
                   std::declval<Exec>(), std::declval<x_t>(), std::declval<sum_of_squares_result<Scalar>>())),
                 sum_of_squares_result<Scalar>>::value
    && !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type
{};

} // end anonymous namespace

template <class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor, class Scalar>
sum_of_squares_result<Scalar> vector_sum_of_squares(
  std::experimental::linalg::impl::inline_exec_t&& /* exec */,
  std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> x,
  sum_of_squares_result<Scalar> init)
{
  using std::abs;

  if (x.extent(0) == 0)
  {
    return init;
  }

  // Rescaling, as in the Reference BLAS DNRM2 implementation, avoids
  // unwarranted overflow or underflow.

  Scalar scale = init.scaling_factor;
  Scalar ssq   = init.scaled_sum_of_squares;
  for (SizeType i = 0; i < x.extent(0); ++i)
  {
    if (abs(x(i)) != 0.0)
    {
      const auto absxi    = abs(x(i));
      const auto quotient = scale / absxi;
      if (scale < absxi)
      {
        ssq   = Scalar(1.0) + ssq * quotient * quotient;
        scale = absxi;
      }
      else
      {
        ssq = ssq + quotient * quotient;
      }
    }
  }

  sum_of_squares_result<Scalar> result;
  result.scaled_sum_of_squares = ssq;
  result.scaling_factor        = scale;
  return result;
}

template <class ExecutionPolicy,
          class ElementType,
          class SizeType,
          ::std::size_t ext0,
          class Layout,
          class Accessor,
          class Scalar>
sum_of_squares_result<Scalar> vector_sum_of_squares(
  ExecutionPolicy&& exec,
  std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> v,
  sum_of_squares_result<Scalar> init)
{
  constexpr bool use_custom =
    is_custom_vector_sum_of_squares_avail<decltype(execpolicy_mapper(exec)), decltype(v), Scalar>::value;

  if constexpr (use_custom)
  {
    return vector_sum_of_squares(execpolicy_mapper(exec), v, init);
  }
  else
  {
    return vector_sum_of_squares(std::experimental::linalg::impl::inline_exec_t(), v, init);
  }
}

template <class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor, class Scalar>
sum_of_squares_result<Scalar> vector_sum_of_squares(
  std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> v,
  sum_of_squares_result<Scalar> init)
{
  return vector_sum_of_squares(std::experimental::linalg::impl::default_exec_t(), v, init);
}

} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std

#endif // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_SUM_OF_SQUARES_HPP_
