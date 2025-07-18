/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _CUDA___MDSPAN_FLATTEN
#define _CUDA___MDSPAN_FLATTEN

#include <cuda/__mdspan/flat_mdspan_view.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

/**
 * @brief Create a flattened view of an `mdspan` that allows efficient random
 * elementwise access.
 *
 * The returned view object supports all the usual iterator semantics.
 *
 * Unfortunately, flattening mdspan into a linear iterator ends up with inefficient code-gen as
 * compilers are unable to untangle the internal state required to make this work. This is not
 * really an "implementation quality" issue so much as a fundamental constraint. In order to
 * implement iterators, you need to solve the problem of mapping a linear index to a
 * N-dimensional point in space. This linearization is done via the following:
 *
 * @code{.cpp}
 * std::array<std::size_t, DIM> point;
 *
 * for (auto dim = DIM; dim-- > 0;) {
 *   point[dim] = index % span.extent(dim);
 *   index /= span.extent(dim);
 * }
 * @endcode
 *
 * The problem are the modulus and div commands. Modern compilers are seemingly unable to hoist
 * those computations out of the loop and vectorize the code. So an equivalent loop over the
 * extents "normally":
 *
 * @code{.cpp}
 * for (std::size_t i = 0; i < span.extent(0); ++i) {
 *   for (std::size_t j = 0; j < span.extent(1); ++j) {
 *     span(i, j) = ...
 *   }
 * }
 * @endcode
 *
 * Will be fully vectorized by optimizers, but the following (which is more or less what
 * this iterator expands to):
 *
 * @code{.cpp}
 * for (std::size_t i = 0; i < PROD(span.extents()...); ++i) {
 *   std::array<std::size_t, DIM> point = delinearize(i);
 *
 *   span(point) = ...
 * }
 * @endcode
 *
 * Defeats all known modern optimizing compilers. Therefore, unless this iterator is truly
 * required, the user is **strongly** encouraged to iterate over their mdspan normally.
 *
 * @param md The mdspan to flatten.
 *
 * @return The flat view.
 */
template <typename _Element, typename _Extent, typename _Layout, typename _Accessor>
[[nodiscard]] constexpr __flat_mdspan_view<_CUDA_VSTD::mdspan<_Element, _Extent, _Layout, _Accessor>>
flatten(_CUDA_VSTD::mdspan<_Element, _Extent, _Layout, _Accessor> md) noexcept
{
  return __mdspan_detail::__flat_mdspan_view<_CUDA_VSTD::mdspan<_Element, _Extent, _Layout, _Accessor>>{
    _CUDA_VSTD::move(md)};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MDSPAN_HOST_DEVICE_ACCESSOR
