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
 *
 * @brief This defines a void data interface useful to implement STF
 * dependencies without actual data (e.g. to enforce task dependencies)
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

#include <cuda/experimental/__stf/graph/graph_data_interface.cuh>
#include <cuda/experimental/__stf/stream/stream_data_interface.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>

namespace cuda::experimental::stf
{

template <typename T>
class shape_of;

class void_interface
{};

/**
 * @brief defines the shape of a void interface
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends shape_of
 */
template <>
class shape_of<void_interface>
{
public:
  shape_of()                = default;
  shape_of(const shape_of&) = default;
  shape_of(const void_interface&)
      : shape_of<void_interface>()
  {}

  /// Mandatory method : defined the total number of elements in the shape
  size_t size() const
  {
    return 0;
  }
};

/**
 * @brief A hash of the matrix
 */
template <>
struct hash<void_interface>
{
  ::std::size_t operator()(void_interface const&) const noexcept
  {
    return 42;
  }
};

} // end namespace cuda::experimental::stf
