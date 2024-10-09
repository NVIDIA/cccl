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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_IDX_ABS_MAX_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_IDX_ABS_MAX_HPP_

namespace std { namespace experimental { inline namespace __p1673_version_0 { namespace linalg {

// begin anonymous namespace
namespace {

template <class Exec, class v_t, class = void>
struct is_custom_idx_abs_max_avail : std::false_type {};

template <class Exec, class v_t>
struct is_custom_idx_abs_max_avail<Exec, v_t,
        std::enable_if_t<
                // FRizzi: maybe should use is_convertible?
                std::is_same<decltype(idx_abs_max(std::declval<Exec>(), std::declval<v_t>())),
                        typename v_t::extents_type::size_type>::value &&
                !linalg::impl::is_inline_exec_v<Exec>>> : std::true_type {};

template <class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor>
SizeType idx_abs_max_default_impl(
        std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> v) {
    using std::abs;
    using magnitude_type = decltype(abs(v(0)));

    if (v.extent(0) == 0) {
        return std::numeric_limits<SizeType>::max();
    }

    SizeType maxInd = 0;
    magnitude_type maxVal = abs(v(0));
    for (SizeType i = 1; i < v.extent(0); ++i) {
        if (maxVal < abs(v(i))) {
            maxVal = abs(v(i));
            maxInd = i;
        }
    }
    return maxInd;  // FIXME check for NaN "never less than" stuff
}

}  // end anonymous namespace

template <class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor>
SizeType idx_abs_max(std::experimental::linalg::impl::inline_exec_t&& /* exec */,
        std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> v) {
    return idx_abs_max_default_impl(v);
}

template <class ExecutionPolicy, class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor>
SizeType idx_abs_max(ExecutionPolicy&& exec,
        std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> v) {
    if (v.extent(0) == 0) {
        return std::numeric_limits<SizeType>::max();
    }

    constexpr bool use_custom = is_custom_idx_abs_max_avail<decltype(execpolicy_mapper(exec)), decltype(v)>::value;

    if constexpr (use_custom) {
        return idx_abs_max(execpolicy_mapper(exec), v);
    } else {
        return idx_abs_max(std::experimental::linalg::impl::inline_exec_t(), v);
    }
}

template <class ElementType, class SizeType, ::std::size_t ext0, class Layout, class Accessor>
SizeType idx_abs_max(
        std::experimental::mdspan<ElementType, std::experimental::extents<SizeType, ext0>, Layout, Accessor> v) {
    return idx_abs_max(std::experimental::linalg::impl::default_exec_t(), v);
}

}}}}  // namespace std::experimental::__p1673_version_0::linalg

#endif  // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_IDX_ABS_MAX_HPP_
