// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cstdint> // uint8_t

#include "../constantAssert.h"
#include "SmemPhase.cuh" // SmemPhase
#include "SmemResourceRaw.cuh" // SmemResource

template <typename T>
struct SmemStage
{
  SmemResourceRaw& mSmemResourceRaw;

  __device__ SmemStage(SmemResourceRaw& smemResourceRaw);
  __device__ ~SmemStage();

  // SmemStage is a non-copyable, non-movable type. It must be passed by (mutable)
  // reference to be useful. The reason is that it in case of an accidental copy
  // or move the destructor is called twice. This leads to double-increment of
  // the stage index and results in deadlock or a hardware fault.
  SmemStage(const SmemStage&)             = delete; // Delete copy constructor
  SmemStage(SmemStage&&)                  = delete; // Delete move constructor
  SmemStage& operator=(const SmemStage&)  = delete; // Delete copy assignment
  SmemStage& operator=(const SmemStage&&) = delete; // Delete move assignment
};

template <typename T>
__device__ SmemStage<T>::SmemStage(SmemResourceRaw& smemResourceRaw)
    : mSmemResourceRaw(smemResourceRaw)
{}

template <typename T>
__device__ SmemStage<T>::~SmemStage()
{
  mSmemResourceRaw.incrementStage();
}
// Helper: Container to expose SmemPhase for structured binding
template <typename T, size_t numPhases>
struct SmemPhaseStructuredBinding
{
  SmemResourceRaw& mSmemResourceRaw;

  template <size_t I>
  __device__ SmemPhase<T> get() const
  {
    return SmemPhase<T>(mSmemResourceRaw, I);
  }
  template <size_t I>
  __device__ SmemPhase<T> get()
  {
    return SmemPhase<T>(mSmemResourceRaw, I);
  }
};
// Tuple protocol specializations
namespace std
{
template <typename T, size_t numPhases>
struct tuple_size<SmemPhaseStructuredBinding<T, numPhases>> : std::integral_constant<size_t, numPhases>
{};

template <typename T, size_t I, size_t numPhases>
struct tuple_element<I, SmemPhaseStructuredBinding<T, numPhases>>
{
  using type = SmemPhase<T>;
};
} // namespace std

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// The binding function
template <size_t numPhases, typename T>
static inline __device__ SmemPhaseStructuredBinding<T, numPhases> bindPhases(SmemStage<T>& smemStage)
{
  constantAssert(smemStage.mSmemResourceRaw.mNumPhases == numPhases,
                 "Number of bound phases must match resource phases.");

  return SmemPhaseStructuredBinding<T, numPhases>{smemStage.mSmemResourceRaw};
}
