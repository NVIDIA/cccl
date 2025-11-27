// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cstdint> // uint8_t

#include "../allocators/SmemAllocator.h" // SmemAllocator
#include "../SyncHandler.h" // SyncHandler
#include "../values.h" // Stages, Elems
#include "SmemResourceRaw.cuh" // SmemResourceRaw
#include "SmemStage.cuh" // SmemStage

template <typename T>
struct SmemResource : SmemResourceRaw
{
  __device__ SmemResource(SmemResourceRaw smemResourceRaw);
  __device__ [[nodiscard]] SmemStage<T> popStage();
};

template <typename T>
__device__ SmemResource<T>::SmemResource(SmemResourceRaw smemResourceRaw)
    : SmemResourceRaw(smemResourceRaw)
{}

template <typename T>
__device__ SmemStage<T> SmemResource<T>::popStage()
{
  return SmemStage<T>(*this);
}

template <typename StageType, int stageCount>
__host__ __device__ SmemResource<StageType>
makeSmemResource(SyncHandler& syncHandler, StageType (&smemBuffer)[stageCount])
{
  int sizeBytes = sizeof(smemBuffer[0]);
  int stride    = sizeof(smemBuffer[0]);

  auto raw = SmemResourceRaw(syncHandler, smemBuffer, sizeBytes, stride, stageCount);
  return SmemResource<StageType>(raw);
}

template <typename StageType>
__host__ __device__ SmemResource<StageType>
makeSmemResource(SyncHandler& syncHandler, SmemAllocator& smemAllocator, Stages stages, Elems elems)
{
  int align       = alignof(StageType);
  int sizeBytes   = elems.value() * sizeof(StageType);
  int strideBytes = sizeBytes;

  void* ptrBase = smemAllocator.alloc(stages.value() * strideBytes, align);
  auto raw      = SmemResourceRaw(syncHandler, ptrBase, sizeBytes, strideBytes, stages.value());
  return SmemResource<StageType>(raw);
}

template <typename StageType>
__host__ __device__ SmemResource<StageType>
makeSmemResource(SyncHandler& syncHandler, SmemAllocator& smemAllocator, Stages stages)
{
  return makeSmemResource<StageType>(syncHandler, smemAllocator, stages, elems(1));
}
