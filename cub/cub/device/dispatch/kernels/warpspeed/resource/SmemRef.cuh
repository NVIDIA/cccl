// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cstdint> // uint8_t

#include "../squad/Squad.h" // Squad
#include "SmemResourceRaw.cuh" // SmemResourceRaw

template <typename T>
struct SmemRef
{
  SmemResourceRaw& mSmemResourceRaw;
  int mCurPhase;
  bool mTxCountIsSet           = false;
  int mTxCount                 = 0;
  bool mDoFenceLdsToAsyncProxy = false;

  __device__ SmemRef(SmemResourceRaw& smemResourceRaw, int phase);
  // SmemRef is a non-copyable, non-movable type. It must be passed by (mutable)
  // reference to be useful. The reason is that it in case of an accidental copy
  // or move the destructor is called twice. This leads to double-arrivals on
  // barriers and results in deadlock or a hardware fault.
  SmemRef(const SmemRef&)             = delete; // Delete copy constructor
  SmemRef(SmemRef&&)                  = delete; // Delete move constructor
  SmemRef& operator=(const SmemRef&)  = delete; // Delete copy assignment
  SmemRef& operator=(const SmemRef&&) = delete; // Delete move assignment

  __device__ T& data();
  __device__ int sizeBytes() const;

  __device__ uint64_t* ptrCurBarrierRelease();

  __device__ void squadIncreaseTxCount(const Squad& squad, int txCount);
  __device__ void setFenceLdsToAsyncProxy();

  __device__ ~SmemRef();
};

template <typename T>
__device__ SmemRef<T>::SmemRef(SmemResourceRaw& smemResourceRaw, int phase)
    : mSmemResourceRaw(smemResourceRaw)
    , mCurPhase(phase)
{}

template <typename T>
__device__ T& SmemRef<T>::data()
{
  return reinterpret_cast<T&>(*(T*) mSmemResourceRaw.data());
}

template <typename T>
__device__ int SmemRef<T>::sizeBytes() const
{
  return mSmemResourceRaw.mSizeBytes;
}

template <typename T>
__device__ uint64_t* SmemRef<T>::ptrCurBarrierRelease()
{
  return mSmemResourceRaw.ptrCurBarrierRelease(mCurPhase);
}

template <typename T>
__device__ void SmemRef<T>::squadIncreaseTxCount(const Squad& squad, int txCount)
{
  mTxCountIsSet = true;
  // Only leader thread increments txCount
  txCount = squad.isLeaderThread() ? txCount : 0;
  mTxCount += txCount;
}

template <typename T>
__device__ void SmemRef<T>::setFenceLdsToAsyncProxy()
{
  mDoFenceLdsToAsyncProxy = true;
}

template <typename T>
__device__ SmemRef<T>::~SmemRef()
{
  if (mDoFenceLdsToAsyncProxy)
  {
    mSmemResourceRaw.fenceLdsToAsyncProxy();
  }
  if (mTxCountIsSet)
  {
    mSmemResourceRaw.releaseTx(mCurPhase, mTxCount);
  }
  else
  {
    mSmemResourceRaw.release(mCurPhase);
  }
}
