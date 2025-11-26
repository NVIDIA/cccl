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
