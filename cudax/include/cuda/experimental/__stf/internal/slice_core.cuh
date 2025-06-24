//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Definition of `slice` and related artifacts
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/mdspan>

namespace cuda::experimental::stf
{
using ::cuda::std::mdspan;
}

namespace cuda::experimental::stf
{

/**
 * @brief Abstraction for the shape of a structure such as a multidimensional view, i.e. everything but the data
 * itself. A type to be used with cudastf must specialize this template.
 *
 * @tparam T The data corresponding to this shape.
 *
 * Any specialization must be default constructible, copyable, and assignable. The required methods are constructor from
 * `const T&`. All other definitions are optional and should provide full information about the structure ("shape") of
 * an object of type `T`, without actually allocating any data for it.
 *
 * @class shape_of
 */
template <typename T>
class shape_of;

#if defined(CUDASTF_BOUNDSCHECK) && defined(NDEBUG)
#  error "CUDASTF_BOUNDSCHECK requires that NDEBUG is not defined."
#endif

/**
 * @brief A layout stride that can be used with `mdspan`.
 *
 * In debug mode (i.e., `NDEBUG` is not defined) all uses of `operator()()` are bounds-checked by means of `assert`.
 */
// struct layout_stride : ::cuda::std::layout_stride
// {
//   template <class Extents>
//   struct mapping : ::cuda::std::layout_stride::mapping<Extents>
//   {
//     constexpr mapping() = default;
// 
//     template <typename... A>
//     constexpr _CCCL_HOST_DEVICE mapping(A&&... a)
//         : ::cuda::std::layout_stride::mapping<Extents>(::std::forward<A>(a)...)
//     {}
// 
//     template <typename... is_t>
//     constexpr _CCCL_HOST_DEVICE auto operator()(is_t&&... is) const
//     {
// #ifdef CUDASTF_BOUNDSCHECK
//       each_in_pack(
//         [&](auto r, const auto& i) {
//           _CCCL_ASSERT(i < this->extents().extent(r), "Index out of bounds.");
//         },
//         is...);
// #endif
//       return ::cuda::std::layout_stride::mapping<Extents>::operator()(::std::forward<is_t>(is)...);
//     }
//   };
// };

using layout_stride = ::cuda::std::layout_stride;

/**
 * @brief Slice based on `mdspan`.
 *
 * @tparam T
 * @tparam dimensions
 */
template <typename T, size_t dimensions = 1>
using slice = mdspan<T, ::cuda::std::dextents<size_t, dimensions>, layout_stride>;

} // namespace cuda::experimental::stf
