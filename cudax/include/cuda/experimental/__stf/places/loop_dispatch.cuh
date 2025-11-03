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
 * @brief Mechanism to dispatch computation over a set of execution places.
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

#include <cuda/experimental/__stf/places/place_partition.cuh>

#include <thread>

namespace cuda::experimental::stf
{
namespace reserved
{
/* A very simple hash function to give the impression of a random distribution of indices */
inline size_t customHash(size_t value)
{
  size_t hash = value;

  // Mix the bits using XOR and bit shifts
  hash ^= (hash << 13);
  hash ^= (hash >> 17);
  hash ^= (hash << 5);

  return hash;
}
} // end namespace reserved

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
/* TODO : introduce a policy to decide whether or not to use threads, and the thread-index mapping (currently random) */
template <typename context_t, typename exec_place_t, bool use_threads = true>
inline void loop_dispatch(
  context_t ctx,
  exec_place_t root_exec_place,
  place_partition_scope scope,
  size_t start,
  size_t end,
  ::std::function<void(size_t)> func)
{
  auto partition = place_partition(ctx.async_resources(), root_exec_place, scope);

  size_t cnt = end - start;

  size_t place_cnt = partition.size();

  size_t nthreads = ::std::min(place_cnt, cnt);

  ::std::vector<::std::thread> threads;

  // loop in reversed order so that tid=0 comes last, and we can execute it
  // in the calling thread while other threads are working
  for (size_t tid = nthreads; tid-- > 0;)
  {
    // Work that should be performed by thread "tid"
    auto tid_work = [=, &func]() {
      // Distribute subplaces in a round robin fashion
      ::std::vector<::std::shared_ptr<exec_place>> thread_affinity;
      for (size_t i = tid; i < place_cnt; i += nthreads)
      {
        thread_affinity.push_back(::std::make_shared<exec_place>(partition.get(tid)));
      }
      ctx.push_affinity(mv(thread_affinity));

      for (size_t i = start; i < end; i++)
      {
        if (reserved::customHash(i) % nthreads == tid)
        {
          func(i);
        }
      }

      ctx.pop_affinity();
    };

    if (!use_threads || tid == 0)
    {
      // This is done by the current thread
      tid_work();
    }
    else
    {
      threads.emplace_back(tid_work);
    }
  }

  // If we don't use threads, check none was created
  assert(use_threads || threads.size() == 0);

  // Wait for all threads to complete
  if (use_threads)
  {
    for (auto& thread : threads)
    {
      thread.join();
    }
  }
}

/*
 * Overload of loop_dispatch which automatically selects the partitioning scope
 */
template <typename context_t, typename exec_place_t, bool use_threads = true>
inline void
loop_dispatch(context_t ctx, exec_place_t root_exec_place, size_t start, size_t end, ::std::function<void(size_t)> func)
{
  // Partition among devices by default
  place_partition_scope scope = place_partition_scope::cuda_device;

  if (getenv("CUDASTF_GREEN_CONTEXT_SIZE"))
  {
    scope = place_partition_scope::green_context;
  }

  loop_dispatch<context_t, exec_place_t, use_threads>(mv(ctx), mv(root_exec_place), scope, start, end, mv(func));
}

/*
 * Overload of loop_dispatch which automatically selects the current affinity
 * based on the ctx (or takes all devices) and selects the scope
 */
template <typename context_t, bool use_threads = true>
inline void loop_dispatch(context_t ctx, size_t start, size_t end, ::std::function<void(size_t)> func)
{
  // Partition among devices by default
  place_partition_scope scope = place_partition_scope::cuda_device;

  if (getenv("CUDASTF_GREEN_CONTEXT_SIZE"))
  {
    scope = place_partition_scope::green_context;
  }

  // The type of the "exec_place_t" differs if we already use a vector of
  // pointers to places, or an actual execution place, so we do not factorize
  // it yet.
  if (ctx.has_affinity())
  {
    loop_dispatch<context_t, ::std::vector<::std::shared_ptr<exec_place>>, use_threads>(
      mv(ctx), ctx.current_affinity(), scope, start, end, mv(func));
  }
  else
  {
    loop_dispatch<context_t, exec_place_grid, use_threads>(
      mv(ctx), exec_place::all_devices(), scope, start, end, mv(func));
  }
}
#endif // _CCCL_DOXYGEN_INVOKED
} // end namespace cuda::experimental::stf
