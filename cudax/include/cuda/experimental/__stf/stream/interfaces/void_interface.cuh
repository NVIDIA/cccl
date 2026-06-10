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
 * @brief This implements the void data interface in the stream_ctx backend
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

#include <cuda/experimental/__stf/internal/void_interface.cuh>
#include <cuda/experimental/__stf/stream/stream_data_interface.cuh>

namespace cuda::experimental::stf
{
template <typename T>
struct streamed_interface_of;

/**
 * @brief Data interface to manipulate the void interface in the CUDA stream backend
 */
class void_stream_interface : public stream_data_interface_simple<void_interface>
{
public:
  using base = stream_data_interface_simple<void_interface>;
  using base::shape_t;

  void_stream_interface(void_interface m)
      : base(::std::move(m))
  {}
  void_stream_interface(typename base::shape_t s)
      : base(s)
  {}

  /// Copy the content of an instance to another instance : this is a no-op
  void stream_data_copy(const data_place&, instance_id_t, const data_place&, instance_id_t, cudaStream_t) override {}

  /// Pretend we allocate an instance on a specific data place : we do not do any allocation here
  void stream_data_allocate(
    backend_ctx_untyped&, const data_place&, instance_id_t, ::std::ptrdiff_t& s, void**, cudaStream_t) override
  {
    // By filling a non negative number, we notify that the allocation was successful
    s = 0;
  }

  /// Pretend we deallocate an instance (no-op)
  void stream_data_deallocate(backend_ctx_untyped&, const data_place&, instance_id_t, void*, cudaStream_t) override {}

  bool pin_host_memory(instance_id_t) override
  {
    // no-op
    return false;
  }

  void unpin_host_memory(instance_id_t) override {}

  /* This helps detecting when we are manipulating a void data interface, so
   * that we can optimize useless stages such as allocations or copies */
  bool is_void_interface() const override final
  {
    return true;
  }
};

/**
 * @brief Define how the CUDA stream backend must manipulate this void interface
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends streamed_interface_of
 */
template <>
struct streamed_interface_of<void_interface>
{
  using type = void_stream_interface;
};
} // end namespace cuda::experimental::stf
