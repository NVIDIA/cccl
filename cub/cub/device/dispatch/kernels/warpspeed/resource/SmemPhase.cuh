/***************************************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
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
