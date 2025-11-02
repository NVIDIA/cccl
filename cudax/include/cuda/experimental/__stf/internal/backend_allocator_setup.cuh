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
 * @brief Implements code that is common to both backends but which needs full definition of backend_ctx_untyped
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

#include <cuda/experimental/__stf/allocators/cached_allocator.cuh>
#include <cuda/experimental/__stf/allocators/pooled_allocator.cuh>

namespace cuda::experimental::stf
{
class block_allocator_untyped;
}

namespace cuda::experimental::stf::reserved
{
template <typename T, typename ctx_impl_t, typename... Args>
auto allocators_create_and_attach(ctx_impl_t& i, Args&&... args)
{
  /* We cannot create a block_allocator<T> directly because the context
   * is not usable yet, this will not attach the allocator. */
  auto res = block_allocator_untyped(::std::make_shared<T>(::std::forward<Args>(args)...));

  /* Ensure the allocator is cleaned up when the context ends */
  i.attached_allocators.push_back(res);

  return res;
};

template <typename ctx_impl_t>
void backend_ctx_set_default_allocator(ctx_impl_t& i, block_allocator_untyped& uncached)
{
  const char* default_alloc_env = getenv("CUDASTF_DEFAULT_ALLOCATOR");
  if (default_alloc_env)
  {
    ::std::string default_alloc_str(default_alloc_env);
    if (default_alloc_str == "uncached")
    {
      i.default_allocator = uncached;
    }
    else if (default_alloc_str == "cached")
    {
      i.default_allocator = allocators_create_and_attach<cached_block_allocator>(i, uncached);
    }
    else if (default_alloc_str == "cached_fifo")
    {
      i.default_allocator = allocators_create_and_attach<cached_block_allocator_fifo>(i, uncached);
    }
    else if (default_alloc_str == "pooled")
    {
      i.default_allocator = allocators_create_and_attach<pooled_allocator>(i);
    }
    else
    {
      fprintf(stderr, "Error: invalid CUDASTF_DEFAULT_ALLOCATOR value.\n");
      abort();
    }
  }
  else
  {
    // Default allocator = cached
    i.default_allocator = allocators_create_and_attach<cached_block_allocator>(i, uncached);
  }

  // Make it possible to customize the context-wide default allocator (use the default one for now)
  i.custom_allocator = i.default_allocator;
}

/**
 * @brief This code selects the appropriate allocators when creating contexts,
 * it was moved to a separate file because the same code is used in different
 * context implementations.
 *
 * This method is a friend of the backend_ctx_untyped class so that it may
 * modify its internal structures.
 */
template <typename ctx_impl_t, typename uncached_allocator_t>
void backend_ctx_setup_allocators(ctx_impl_t& i)
{
  i.uncached_allocator = allocators_create_and_attach<uncached_allocator_t>(i);

  // Select the default allocator based on this uncached allocator
  backend_ctx_set_default_allocator(i, i.uncached_allocator);
}

/**
 * @brief Select a different uncached allocator after the context was already
 *        created. We change the default allocators too to reflect this new uncached
 *        allocator.
 *
 * Note : this resets the current context allocator (set_allocator) to the
 *        default one too.
 */
template <typename ctx_impl_t>
void backend_ctx_update_uncached_allocator(ctx_impl_t& i, block_allocator_untyped uncached_allocator)
{
  // Update the uncached allocator
  i.uncached_allocator = uncached_allocator;

  backend_ctx_set_default_allocator(i, uncached_allocator);
}
} // namespace cuda::experimental::stf::reserved
