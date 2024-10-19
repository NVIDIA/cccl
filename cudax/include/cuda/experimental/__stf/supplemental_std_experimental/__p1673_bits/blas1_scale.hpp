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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_SCALE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_SCALE_HPP_

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

template <class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor, class Scalar>
void linalg_scale_rank_1(
  const Scalar alpha,
  std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> x)
{
  for (SizeType i = 0; i < x.extent(0); ++i)
  {
    x(i) *= alpha;
  }
}

template <class ElementType,
          class SizeType,
          ::std::size_t numRows,
          ::std::size_t numCols,
          class Layout,
          class Accessor,
          class Scalar>
void linalg_scale_rank_2(
  const Scalar alpha,
  std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, numRows, numCols>, Layout, Accessor> A)
{
  for (SizeType j = 0; j < A.extent(1); ++j)
  {
    for (SizeType i = 0; i < A.extent(0); ++i)
    {
      A(i, j) *= alpha;
    }
  }
}

template <typename Exec, typename Scalar, typename x_t, typename = void>
struct is_custom_scale_avail : std::false_type
{};

template <typename Exec, typename Scalar, typename x_t>
struct is_custom_scale_avail<
  Exec,
  Scalar,
  x_t,
  std::enable_if_t<std::is_void_v<decltype(scale(std::declval<Exec>(), std::declval<Scalar>(), std::declval<x_t>()))>
                   && !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type
{};

} // end anonymous namespace

template <class Scalar, class ElementType, class SizeType, ::std::size_t... ext, class Layout, class Accessor>
void scale(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
           const Scalar alpha,
           std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext...>, Layout, Accessor> x)
{
  static_assert(x.rank() <= 2);

  if constexpr (x.rank() == 1)
  {
    linalg_scale_rank_1(alpha, x);
  }
  else if constexpr (x.rank() == 2)
  {
    linalg_scale_rank_2(alpha, x);
  }
}

template <class ExecutionPolicy,
          class Scalar,
          class ElementType,
          class SizeType,
          ::std::size_t... ext,
          class Layout,
          class Accessor>
void scale(ExecutionPolicy&& exec,
           const Scalar alpha,
           std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext...>, Layout, Accessor> x)
{
  // Call custom overload if available else call std implementation

  constexpr bool use_custom =
    is_custom_scale_avail<decltype(execpolicy_mapper(exec)), decltype(alpha), decltype(x)>::value;

  if constexpr (use_custom)
  {
    scale(execpolicy_mapper(exec), alpha, x);
  }
  else
  {
    scale(std::experimental::linalg::impl::inline_exec_t(), alpha, x);
  }
}

template <class Scalar, class ElementType, class SizeType, ::std::size_t... ext, class Layout, class Accessor>
void scale(const Scalar alpha,
           std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext...>, Layout, Accessor> x)
{
  scale(std::experimental::linalg::impl::default_exec_t(), alpha, x);
}

} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std

#endif // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_SCALE_HPP_
