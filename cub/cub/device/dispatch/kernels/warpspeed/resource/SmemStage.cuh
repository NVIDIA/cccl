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

#include <cub/device/dispatch/kernels/warpspeed/constantAssert.h>
#include <cub/device/dispatch/kernels/warpspeed/SmemPhase.cuh>
#include <cub/device/dispatch/kernels/warpspeed/SmemResourceRaw.cuh>

#include <cuda/std/__tuple_dir/structured_bindings.h>
#include <cuda/std/cstdint>

template <typename T>
struct SmemStage
{
  SmemResourceRaw& mSmemResourceRaw;

  _CCCL_DEVICE_API SmemStage(SmemResourceRaw& smemResourceRaw) noexcept
      : mSmemResourceRaw(smemResourceRaw)
  {}

  _CCCL_DEVICE_API ~SmemStage()
  {
    mSmemResourceRaw.incrementStage();
  }

  // SmemStage is a non-copyable, non-movable type. It must be passed by (mutable)
  // reference to be useful. The reason is that it in case of an accidental copy
  // or move the destructor is called twice. This leads to double-increment of
  // the stage index and results in deadlock or a hardware fault.
  SmemStage(const SmemStage&)             = delete; // Delete copy constructor
  SmemStage(SmemStage&&)                  = delete; // Delete move constructor
  SmemStage& operator=(const SmemStage&)  = delete; // Delete copy assignment
  SmemStage& operator=(const SmemStage&&) = delete; // Delete move assignment
};

// Helper: Container to expose SmemPhase for structured binding
template <typename T, size_t numPhases>
struct SmemPhaseStructuredBinding
{
  SmemResourceRaw& mSmemResourceRaw;

  template <size_t I>
  [[nodiscard]] _CCCL_DEVICE_API SmemPhase<T> get() const
  {
    return SmemPhase<T>(mSmemResourceRaw, I);
  }
  template <size_t I>
  [[nodiscard]] _CCCL_DEVICE_API SmemPhase<T> get()
  {
    return SmemPhase<T>(mSmemResourceRaw, I);
  }
};
// Tuple protocol specializations
namespace std
{
template <typename T, size_t numPhases>
struct tuple_size<SmemPhaseStructuredBinding<T, numPhases>>
{
  static constexpr size_t value = numPhases;
};

template <typename T, size_t I, size_t numPhases>
struct tuple_element<I, SmemPhaseStructuredBinding<T, numPhases>>
{
  using type = SmemPhase<T>;
};
} // namespace std

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// The binding function
template <size_t numPhases, typename T>
static [[nodiscard]] _CCCL_DEVICE_API SmemPhaseStructuredBinding<T, numPhases> bindPhases(SmemStage<T>& smemStage)
{
  constantAssert(smemStage.mSmemResourceRaw.mNumPhases == numPhases,
                 "Number of bound phases must match resource phases.");

  return SmemPhaseStructuredBinding<T, numPhases>{smemStage.mSmemResourceRaw};
}
