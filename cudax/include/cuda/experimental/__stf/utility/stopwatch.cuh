//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime.h>

#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

#include <chrono>
#include <thread>

namespace cuda::experimental::stf
{

class stopwatch
{
public:
  stopwatch(const stopwatch&)            = delete;
  stopwatch& operator=(const stopwatch&) = delete;

  stopwatch(stopwatch&& rhs) noexcept
      : _start(rhs._start)
      , _stop(rhs._stop)
      , _state(rhs._state)
  {
    rhs._start = rhs._stop = nullptr;
    rhs._state             = state::invalid;
  }

  stopwatch& operator=(stopwatch&& rhs)
  {
    ::std::swap(_start, rhs._start);
    ::std::swap(_stop, rhs._stop);
    ::std::swap(_state, rhs._state);
    return *this;
  }

  stopwatch()
      : _state(state::idle)
  {
    cuda_safe_call(cudaEventCreate(&_start));
    cuda_safe_call(cudaEventCreate(&_stop));
  }

  static inline constexpr class autostart_t
  {
  } autostart = {};

  stopwatch(autostart_t, cudaStream_t stream = nullptr)
      : stopwatch()
  {
    start(stream);
  }

  ~stopwatch()
  {
    if (_state != state::invalid)
    {
      cuda_safe_call(cudaEventDestroy(_start));
      cuda_safe_call(cudaEventDestroy(_stop));
    }
  }

  void start(cudaStream_t stream = nullptr)
  {
    assert(_state == state::idle || _state == state::stopped);
    cuda_safe_call(cudaEventRecord(_start, stream));
    _state = state::started;
  }

  void stop(cudaStream_t stream = nullptr)
  {
    assert(_state == state::started);
    cuda_safe_call(cudaEventRecord(_stop, stream));
    _state = state::stopped;
  }

  ::std::chrono::duration<float, ::std::milli> elapsed()
  {
    assert(state::stopped);
    float result;
    cuda_safe_call(cudaEventSynchronize(_stop));
    cuda_safe_call(cudaEventElapsedTime(&result, _start, _stop));
    return ::std::chrono::duration<float, ::std::milli>(result);
  }

  template <typename T = char>
  float bandwidth(size_t items_transferred)
  {
    return 1000. * sizeof(T) * items_transferred / elapsed().count();
  }

private:
  cudaEvent_t _start;
  cudaEvent_t _stop;
  enum class state
  {
    invalid,
    idle,
    started,
    stopped
  } _state = state::invalid;
};

#ifdef UNITTESTED_FILE
UNITTEST("stopwatch")
{
  stopwatch sw(stopwatch::autostart);
  ::std::this_thread::sleep_for(::std::chrono::milliseconds(100));
  sw.stop();
  auto elapsed = sw.elapsed();
  assert(elapsed.count() >= 100);
};
#endif // UNITTESTED_FILE

} // namespace cuda::experimental::stf
