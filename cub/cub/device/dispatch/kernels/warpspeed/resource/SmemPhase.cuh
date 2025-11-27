// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/kernels/warpspeed/SmemRef.cuh> // SmemRef
#include <cub/device/dispatch/kernels/warpspeed/SmemResourceRaw.cuh> // SmemResourceRaw

#include <cuda/std/cstdint> // uint8_t


template <typename T>
struct SmemPhase
{
  SmemResourceRaw& mSmemResourceRaw;
  int mCurPhase;

  _CCCL_DEVICE_API SmemPhase(SmemResourceRaw& smemResourceRaw, int phase);
  _CCCL_DEVICE_API SmemRef<T> acquireRef();
};

template <typename T>
_CCCL_DEVICE_API SmemPhase<T>::SmemPhase(SmemResourceRaw& smemResourceRaw, int phase)
    : mSmemResourceRaw(smemResourceRaw)
    , mCurPhase(phase)
{}

template <typename T>
_CCCL_DEVICE_API SmemRef<T> SmemPhase<T>::acquireRef()
{
  // Wait on barrier
  mSmemResourceRaw.acquire(mCurPhase);
  // Return ref
  return SmemRef<T>(mSmemResourceRaw, mCurPhase);
}
