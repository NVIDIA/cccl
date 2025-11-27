// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cstdint> // uint8_t

#include "SmemRef.cuh" // SmemRef
#include "SmemResourceRaw.cuh" // SmemResourceRaw

template <typename T>
struct SmemPhase
{
  SmemResourceRaw& mSmemResourceRaw;
  int mCurPhase;

  __device__ SmemPhase(SmemResourceRaw& smemResourceRaw, int phase);
  __device__ SmemRef<T> acquireRef();
};

template <typename T>
__device__ SmemPhase<T>::SmemPhase(SmemResourceRaw& smemResourceRaw, int phase)
    : mSmemResourceRaw(smemResourceRaw)
    , mCurPhase(phase)
{}

template <typename T>
__device__ SmemRef<T> SmemPhase<T>::acquireRef()
{
  // Wait on barrier
  mSmemResourceRaw.acquire(mCurPhase);
  // Return ref
  return SmemRef<T>(mSmemResourceRaw, mCurPhase);
}
