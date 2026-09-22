// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/warpspeed/constant_assert.cuh>
#include <cub/detail/warpspeed/resource/smem_phase.cuh>
#include <cub/detail/warpspeed/resource/smem_resource_raw.cuh>

#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
template <typename Tp>
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
template <typename Tp, ::cuda::std::size_t NumPhases>
struct SmemPhaseStructuredBinding
{
  SmemResourceRaw& mSmemResourceRaw;

  template <::cuda::std::size_t Index>
  [[nodiscard]] _CCCL_DEVICE_API SmemPhase<Tp> get() const
  {
    return SmemPhase<Tp>(mSmemResourceRaw, Index);
  }
};

// The binding function
template <::cuda::std::size_t NumPhases, typename Tp>
[[nodiscard]] _CCCL_DEVICE_API SmemPhaseStructuredBinding<Tp, NumPhases> bindPhases(SmemStage<Tp>& smemStage)
{
  _WS_CONSTANT_ASSERT(smemStage.mSmemResourceRaw.mNumPhases == NumPhases,
                      "Number of bound phases must match resource phases.");

  return SmemPhaseStructuredBinding<Tp, NumPhases>{smemStage.mSmemResourceRaw};
}
} // namespace detail::warpspeed

CUB_NAMESPACE_END

// Tuple protocol specializations
namespace std
{
template <typename Tp, size_t NumPhases>
struct tuple_size<CUB_NS_QUALIFIER::detail::warpspeed::SmemPhaseStructuredBinding<Tp, NumPhases>>
{
  static constexpr size_t value = NumPhases;
};

template <typename Tp, size_t Index, ::cuda::std::size_t NumPhases>
struct tuple_element<Index, CUB_NS_QUALIFIER::detail::warpspeed::SmemPhaseStructuredBinding<Tp, NumPhases>>
{
  using type = CUB_NS_QUALIFIER::detail::warpspeed::SmemPhase<Tp>;
};
} // namespace std
