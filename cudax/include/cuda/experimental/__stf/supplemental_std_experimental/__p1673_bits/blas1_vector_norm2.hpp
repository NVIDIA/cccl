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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_NORM2_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_NORM2_HPP_

#include "blas1_vector_sum_of_squares.hpp"

#include <cmath>
#include <cstdlib>

namespace std { namespace experimental { inline namespace __p1673_version_0 { namespace linalg {

namespace {
template <class Exec, class x_t, class Scalar, class = void>
struct is_custom_vector_norm2_avail : std::false_type {};

template <class Exec, class x_t, class Scalar>
struct is_custom_vector_norm2_avail<Exec, x_t, Scalar,
        std::enable_if_t<
                std::is_same<decltype(vector_norm2(std::declval<Exec>(), std::declval<x_t>(), std::declval<Scalar>())),
                        Scalar>::value &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};
}  // end anonymous namespace

template <class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor, class Scalar>
Scalar vector_norm2(std::experimental::linalg::impl::inline_exec_t&& exec,
        std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> x,
        Scalar init) {
    // Initialize the sum of squares result
    sum_of_squares_result<Scalar> ssq_init;
    ssq_init.scaling_factor = Scalar {};
    // FIXME (Hoemmen 2021/05/27) We'll need separate versions of this
    // for types whose "one" we don't know how to construct.
    ssq_init.scaled_sum_of_squares = 1.0;

    // Compute the sum of squares using an algorithm that avoids
    // underflow and overflow by scaling.
    auto ssq_res = vector_sum_of_squares(exec, x, ssq_init);
    using std::sqrt;
    return init + ssq_res.scaling_factor * sqrt(ssq_res.scaled_sum_of_squares);
}

template <class ExecutionPolicy, class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor,
        class Scalar>
Scalar vector_norm2(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> x,
        Scalar init) {
    constexpr bool use_custom =
            is_custom_vector_norm2_avail<decltype(execpolicy_mapper(exec)), decltype(x), Scalar>::value;

    if constexpr (use_custom) {
        return vector_norm2(execpolicy_mapper(exec), x, init);
    } else {
        return vector_norm2(std::experimental::linalg::impl::inline_exec_t(), x, init);
    }
}

template <class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor, class Scalar>
Scalar vector_norm2(
        std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> x,
        Scalar init) {
    return vector_norm2(std::experimental::linalg::impl::default_exec_t(), x, init);
}

namespace vector_norm2_detail {
using std::abs;

// The point of this is to do correct ADL for abs,
// without exposing "using std::abs" in the outer namespace.
template <class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor>
auto vector_norm2_return_type_deducer(
        std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> x)
        -> decltype(abs(x(0)) * abs(x(0)));
}  // namespace vector_norm2_detail

template <class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor>
auto vector_norm2(
        std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> x)
        -> decltype(vector_norm2_detail::vector_norm2_return_type_deducer(x)) {
    using return_t = decltype(vector_norm2_detail::vector_norm2_return_type_deducer(x));
    return vector_norm2(x, return_t {});
}

template <class ExecutionPolicy, class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor>
auto vector_norm2(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> x)
        -> decltype(vector_norm2_detail::vector_norm2_return_type_deducer(x)) {
    using return_t = decltype(vector_norm2_detail::vector_norm2_return_type_deducer(x));
    return vector_norm2(exec, x, return_t {});
}

}}}}  // namespace std::experimental::__p1673_version_0::linalg

#endif  // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_NORM2_HPP_
