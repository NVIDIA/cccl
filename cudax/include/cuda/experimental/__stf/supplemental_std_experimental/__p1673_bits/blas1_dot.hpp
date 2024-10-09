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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_DOT_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_DOT_HPP_

#include <type_traits>

namespace std { namespace experimental { inline namespace __p1673_version_0 { namespace linalg {

// begin anonymous namespace
namespace {

template <class Exec, class v1_t, class v2_t, class Scalar, class = void>
struct is_custom_dot_avail : std::false_type {};

template <class Exec, class v1_t, class v2_t, class Scalar>
struct is_custom_dot_avail<Exec, v1_t, v2_t, Scalar,
        std::enable_if_t<std::is_same<decltype(dot(std::declval<Exec>(), std::declval<v1_t>(), std::declval<v2_t>(),
                                              std::declval<Scalar>())),
                                 Scalar>::value &&
                         !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

}  // end anonymous namespace

template <class ElementType1, class SizeType1, ::std::size_t ext1, class Layout1, class Accessor1, class ElementType2,
        class SizeType2, ::std::size_t ext2, class Layout2, class Accessor2, class Scalar>
Scalar dot(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType1, std::experimental::extents<SizeType1, ext1>, Layout1, Accessor1> v1,
        std::experimental::mdspan<ElementType2, std::experimental::extents<SizeType2, ext2>, Layout2, Accessor2> v2,
        Scalar init) {
    static_assert(v1.static_extent(0) == dynamic_extent || v2.static_extent(0) == dynamic_extent ||
                  v1.static_extent(0) == v2.static_extent(0));

    using size_type = std::common_type_t<SizeType1, SizeType2>;
    for (size_type k = 0; k < v1.extent(0); ++k) {
        init += v1(k) * v2(k);
    }
    return init;
}

template <class ExecutionPolicy, class ElementType1, class SizeType1, ::std::size_t ext1, class Layout1,
        class Accessor1, class ElementType2, class SizeType2, ::std::size_t ext2, class Layout2, class Accessor2,
        class Scalar>
Scalar dot(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType1, std::experimental::extents<SizeType1, ext1>, Layout1, Accessor1> v1,
        std::experimental::mdspan<ElementType2, std::experimental::extents<SizeType2, ext2>, Layout2, Accessor2> v2,
        Scalar init) {
    static_assert(v1.static_extent(0) == dynamic_extent || v2.static_extent(0) == dynamic_extent ||
                  v1.static_extent(0) == v2.static_extent(0));

    constexpr bool use_custom =
            is_custom_dot_avail<decltype(execpolicy_mapper(exec)), decltype(v1), decltype(v2), Scalar>::value;

    if constexpr (use_custom) {
        return dot(execpolicy_mapper(exec), v1, v2, init);
    } else {
        return dot(std::experimental::linalg::impl::inline_exec_t(), v1, v2, init);
    }
}

template <class ElementType1, class SizeType1, ::std::size_t ext1, class Layout1, class Accessor1, class ElementType2,
        class SizeType2, ::std::size_t ext2, class Layout2, class Accessor2, class Scalar>
Scalar dot(std::experimental::mdspan<ElementType1, std::experimental::extents<SizeType1, ext1>, Layout1, Accessor1> v1,
        std::experimental::mdspan<ElementType2, std::experimental::extents<SizeType2, ext2>, Layout2, Accessor2> v2,
        Scalar init) {
    return dot(std::experimental::linalg::impl::default_exec_t(), v1, v2, init);
}

template <class ElementType1, class SizeType1, ::std::size_t ext1, class Layout1, class Accessor1, class ElementType2,
        class SizeType2, ::std::size_t ext2, class Layout2, class Accessor2, class Scalar>
Scalar dotc(std::experimental::mdspan<ElementType1, std::experimental::extents<SizeType1, ext1>, Layout1, Accessor1> v1,
        std::experimental::mdspan<ElementType2, std::experimental::extents<SizeType2, ext2>, Layout2, Accessor2> v2,
        Scalar init) {
    return dot(conjugated(v1), v2, init);
}

template <class ExecutionPolicy, class ElementType1, class SizeType1, ::std::size_t ext1, class Layout1,
        class Accessor1, class ElementType2, class SizeType2, ::std::size_t ext2, class Layout2, class Accessor2,
        class Scalar>
Scalar dotc(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType1, std::experimental::extents<SizeType1, ext1>, Layout1, Accessor1> v1,
        std::experimental::mdspan<ElementType2, std::experimental::extents<SizeType2, ext2>, Layout2, Accessor2> v2,
        Scalar init) {
    return dot(exec, conjugated(v1), v2, init);
}

namespace dot_detail {
using std::abs;

// The point of this is to do correct ADL for abs,
// without exposing "using std::abs" in the outer namespace.
template <class ElementType1, class SizeType1, ::std::size_t ext1, class Layout1, class Accessor1, class ElementType2,
        class SizeType2, ::std::size_t ext2, class Layout2, class Accessor2>
auto dot_return_type_deducer(
        std::experimental::mdspan<ElementType1, std::experimental::extents<SizeType1, ext1>, Layout1, Accessor1> x,
        std::experimental::mdspan<ElementType2, std::experimental::extents<SizeType2, ext2>, Layout2, Accessor2> y)
        -> decltype(x(0) * y(0));
}  // namespace dot_detail

template <class ElementType1, class SizeType1, ::std::size_t ext1, class Layout1, class Accessor1, class ElementType2,
        class SizeType2, ::std::size_t ext2, class Layout2, class Accessor2>
auto dot(std::experimental::mdspan<ElementType1, std::experimental::extents<SizeType1, ext1>, Layout1, Accessor1> v1,
        std::experimental::mdspan<ElementType2, std::experimental::extents<SizeType2, ext2>, Layout2, Accessor2> v2)
        -> decltype(dot_detail::dot_return_type_deducer(v1, v2)) {
    using return_t = decltype(dot_detail::dot_return_type_deducer(v1, v2));
    return dot(v1, v2, return_t {});
}

template <class ExecutionPolicy, class ElementType1, class SizeType1, ::std::size_t ext1, class Layout1,
        class Accessor1, class ElementType2, class SizeType2, ::std::size_t ext2, class Layout2, class Accessor2>
auto dot(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType1, std::experimental::extents<SizeType1, ext1>, Layout1, Accessor1> v1,
        std::experimental::mdspan<ElementType2, std::experimental::extents<SizeType2, ext2>, Layout2, Accessor2> v2)
        -> decltype(dot_detail::dot_return_type_deducer(v1, v2)) {
    using return_t = decltype(dot_detail::dot_return_type_deducer(v1, v2));
    return dot(exec, v1, v2, return_t {});
}

template <class ElementType1, class SizeType1, ::std::size_t ext1, class Layout1, class Accessor1, class ElementType2,
        class SizeType2, ::std::size_t ext2, class Layout2, class Accessor2>
auto dotc(std::experimental::mdspan<ElementType1, std::experimental::extents<SizeType1, ext1>, Layout1, Accessor1> v1,
        std::experimental::mdspan<ElementType2, std::experimental::extents<SizeType2, ext2>, Layout2, Accessor2> v2)
        -> decltype(dot_detail::dot_return_type_deducer(conjugated(v1), v2)) {
    using return_t = decltype(dot_detail::dot_return_type_deducer(conjugated(v1), v2));
    return dotc(v1, v2, return_t {});
}

template <class ExecutionPolicy, class ElementType1, class SizeType1, ::std::size_t ext1, class Layout1,
        class Accessor1, class ElementType2, class SizeType2, ::std::size_t ext2, class Layout2, class Accessor2>
auto dotc(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType1, std::experimental::extents<SizeType1, ext1>, Layout1, Accessor1> v1,
        std::experimental::mdspan<ElementType2, std::experimental::extents<SizeType2, ext2>, Layout2, Accessor2> v2)
        -> decltype(dot_detail::dot_return_type_deducer(conjugated(v1), v2)) {
    using return_t = decltype(dot_detail::dot_return_type_deducer(conjugated(v1), v2));
    return dotc(exec, v1, v2, return_t {});
}

}}}}  // namespace std::experimental::__p1673_version_0::linalg

#endif  // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_DOT_HPP_
