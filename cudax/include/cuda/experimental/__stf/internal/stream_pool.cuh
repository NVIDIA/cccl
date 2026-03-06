//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Stream pool and decorated stream types used by places.
 *
 * Definitions of stream_pool::next() and usage of exec_place live in places.cuh
 * (because we need to activate the place to create a stream in the appropriate
 * CUDA context).
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif

#include <cuda/experimental/__stf/places/decorated_stream.cuh>

#include <mutex>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::stf
{
class exec_place;

/**
 * @brief A stream_pool object stores a set of streams associated to a specific
 * CUDA context (device, green context, ...)
 *
 * When a slot is empty, next(place) activates the place (RAII guard) and calls
 * place.create_stream(). Defined in places.cuh so the pool can use exec_place_guard
 * and exec_place::create_stream().
 */
struct stream_pool
{
  stream_pool()  = default;
  ~stream_pool() = default;

  /**
   * @brief stream_pool constructor taking a number of slots.
   *
   * Streams are created lazily only via next(place), which activates the place and calls place.create_stream().
   */
  explicit stream_pool(size_t n)
      : payload(n, decorated_stream(nullptr, k_no_stream_id, -1))
  {}

  stream_pool(stream_pool&& rhs)
  {
    ::std::lock_guard<::std::mutex> locker(rhs.mtx);
    payload = ::std::move(rhs.payload);
    rhs.payload.clear();
    index = rhs.index;
  }

  /**
   * @brief Get the next stream in the pool; when a slot is empty, activate the place (RAII guard) and call
   * place.create_stream(). Defined in places.cuh.
   */
  decorated_stream next(const exec_place& place);

  using iterator = ::std::vector<decorated_stream>::iterator;
  iterator begin()
  {
    return payload.begin();
  }
  iterator end()
  {
    return payload.end();
  }

  size_t size() const
  {
    ::std::lock_guard<::std::mutex> locker(mtx);
    return payload.size();
  }

  mutable ::std::mutex mtx;
  ::std::vector<decorated_stream> payload;
  size_t index = 0;
};
} // namespace cuda::experimental::stf
