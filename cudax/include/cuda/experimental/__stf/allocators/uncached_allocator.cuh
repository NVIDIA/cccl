//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief Uncached allocation strategy where every allocation/deallocation results in a direct API call
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

#include <cuda/experimental/__stf/allocators/block_allocator.cuh>
#include <cuda/experimental/__stf/internal/backend_ctx.cuh>

#include <mutex>

namespace cuda::experimental::stf
{
/**
 * @brief Uncached block allocator where allocations and deallocations are
 * directly performed as CUDA API calls (cudaMallocAsync, ...)
 *
 * This is actually a wrapper on top of the uncached allocator automatically
 * created within each context backend, so deinit is a no-op. The intent of
 * this wrapper is to make it possible to create an allocator policy using this
 * internal allocator.
 */
class uncached_block_allocator : public block_allocator_interface
{
  void*
  allocate(backend_ctx_untyped& ctx, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& prereqs) override
  {
    auto& uncached = ctx.get_uncached_allocator();
    return uncached.allocate(ctx, memory_node, s, prereqs);
  }

  void deallocate(
    backend_ctx_untyped& ctx, const data_place& memory_node, event_list& prereqs, void* ptr, size_t sz) override
  {
    auto& uncached = ctx.get_uncached_allocator();
    uncached.deallocate(ctx, memory_node, prereqs, ptr, sz);
  }

  /* Nothing is done here because this is just a wrapper on top of some
   * existing allocator automatically created during backend initialization
   * */
  event_list deinit(backend_ctx_untyped&) override
  {
    return event_list();
  }

  ::std::string to_string() const override
  {
    return "uncached block allocator (wrapper)";
  }
};
} // end namespace cuda::experimental::stf
