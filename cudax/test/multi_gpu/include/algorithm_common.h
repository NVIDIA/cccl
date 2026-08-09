//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/buffer>
#include <cuda/std/cstddef>

#include <exception>
#include <future>
#include <vector>

// One output iterator per local output buffer. Collected after `out` is fully built so the
// iterators do not dangle across reallocations.
template <class T>
[[nodiscard]] std::vector<typename cuda::device_buffer<T>::iterator>
make_output_iterators(std::vector<cuda::device_buffer<T>>& out)
{
  std::vector<typename cuda::device_buffer<T>::iterator> outputs;

  outputs.reserve(out.size());
  for (auto& buf : out)
  {
    outputs.push_back(buf.begin());
  }
  return outputs;
}

template <class Fn>
void run_threaded(cuda::std::size_t num_ranks, Fn fn)
{
  // Every rank must be launched before any is waited on: the single-communicator `reduce`
  // blocks on a collective, so calling `get()` on rank 0's future before rank 1 is even
  // started would deadlock. Launch all futures into the vector first, then drain them.
  std::vector<std::future<void>> futures;

  futures.reserve(num_ranks);
  for (cuda::std::size_t i = 0; i < num_ranks; ++i)
  {
    futures.push_back(std::async(std::launch::async, fn, i));
  }

  // `std::async` stashes any exception thrown by `fn` in the future and `get()` rethrows it on
  // the main thread, where Catch2 can report it as a normal failure. Any not-yet-drained
  // future still joins its thread in its destructor, so a throw here never leaves a peer
  // waiting on an unposted collective. Drain every future so a failure on rank 0 does not mask
  // one on a peer.
  std::exception_ptr error = nullptr;

  for (auto& f : futures)
  {
    try
    {
      f.get();
    }
    catch (...)
    {
      if (!error)
      {
        error = std::current_exception();
      }
    }
  }

  if (error)
  {
    std::rethrow_exception(error);
  }
}
