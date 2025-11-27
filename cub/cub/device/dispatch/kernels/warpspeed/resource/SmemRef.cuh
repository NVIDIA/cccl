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

#include <cub/device/dispatch/kernels/warpspeed/SmemResourceRaw.cuh> // SmemResourceRaw
#include <cub/device/dispatch/kernels/warpspeed/squad/Squad.h> // Squad

#include <cuda/std/cstdint> // uint8_t

template <typename T>
struct SmemRef
{
  SmemResourceRaw& mSmemResourceRaw;
  int mCurPhase;
  bool mTxCountIsSet           = false;
  int mTxCount                 = 0;
  bool mDoFenceLdsToAsyncProxy = false;

  _CCCL_DEVICE_API SmemRef(SmemResourceRaw& smemResourceRaw, int phase);
  // SmemRef is a non-copyable, non-movable type. It must be passed by (mutable)
  // reference to be useful. The reason is that it in case of an accidental copy
  // or move the destructor is called twice. This leads to double-arrivals on
  // barriers and results in deadlock or a hardware fault.
  SmemRef(const SmemRef&)             = delete; // Delete copy constructor
  SmemRef(SmemRef&&)                  = delete; // Delete move constructor
  SmemRef& operator=(const SmemRef&)  = delete; // Delete copy assignment
  SmemRef& operator=(const SmemRef&&) = delete; // Delete move assignment

  _CCCL_DEVICE_API T& data();
  _CCCL_DEVICE_API int sizeBytes() const;

  _CCCL_DEVICE_API uint64_t* ptrCurBarrierRelease();

  _CCCL_DEVICE_API void squadIncreaseTxCount(const Squad& squad, int txCount);
  _CCCL_DEVICE_API void setFenceLdsToAsyncProxy();

  _CCCL_DEVICE_API ~SmemRef();
};

template <typename T>
_CCCL_DEVICE_API SmemRef<T>::SmemRef(SmemResourceRaw& smemResourceRaw, int phase)
    : mSmemResourceRaw(smemResourceRaw)
    , mCurPhase(phase)
{}

template <typename T>
_CCCL_DEVICE_API T& SmemRef<T>::data()
{
  return reinterpret_cast<T&>(*(T*) mSmemResourceRaw.data());
}

template <typename T>
_CCCL_DEVICE_API int SmemRef<T>::sizeBytes() const
{
  return mSmemResourceRaw.mSizeBytes;
}

template <typename T>
_CCCL_DEVICE_API uint64_t* SmemRef<T>::ptrCurBarrierRelease()
{
  return mSmemResourceRaw.ptrCurBarrierRelease(mCurPhase);
}

template <typename T>
_CCCL_DEVICE_API void SmemRef<T>::squadIncreaseTxCount(const Squad& squad, int txCount)
{
  mTxCountIsSet = true;
  // Only leader thread increments txCount
  txCount = squad.isLeaderThread() ? txCount : 0;
  mTxCount += txCount;
}

template <typename T>
_CCCL_DEVICE_API void SmemRef<T>::setFenceLdsToAsyncProxy()
{
  mDoFenceLdsToAsyncProxy = true;
}

template <typename T>
_CCCL_DEVICE_API SmemRef<T>::~SmemRef()
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
