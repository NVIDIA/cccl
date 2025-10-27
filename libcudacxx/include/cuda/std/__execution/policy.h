//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___EXECUTION_POLICY_H
#define _CUDA_STD___EXECUTION_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__execution/stream_policy.h>
#include <cuda/std/__fwd/policy.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_EXECUTION

//! @brief Enumerates the standard execution policies
enum class __execution_policy : uint8_t
{
  __invalid_execution_policy = 0,
  __sequenced                = 1 << 0,
  __parallel                 = 1 << 1,
  __unsequenced              = 1 << 2,
  __parallel_unsequenced     = __execution_policy::__parallel | __execution_policy::__unsequenced,
};

//! @brief Enumerates the different backends we support
//! @note Not an enum class because a user might specify multiple backends
enum __execution_backend : uint8_t
{
  // The backends we provide
  __backend_none = 0,
#if _CCCL_HAS_BACKEND_CUDA()
  __backend_cuda = 1 << 1,
#endif // _CCCL_HAS_BACKEND_CUDA()
#if _CCCL_HAS_BACKEND_OMP()
  __backend_omp = 1 << 2,
#endif // _CCCL_HAS_BACKEND_OMP()
#if _CCCL_HAS_BACKEND_TBB()
  __backend_tbb = 1 << 3,
#endif // _CCCL_HAS_BACKEND_TBB()
};

[[nodiscard]] _CCCL_API constexpr bool __has_unique_backend(const __execution_backend __backends) noexcept
{
  return ::cuda::std::has_single_bit(static_cast<uint32_t>(__backends));
}

[[nodiscard]] _CCCL_API constexpr bool
__has_matching_backend(const __execution_backend __backends, const __execution_backend __target_backend) noexcept
{
  return (static_cast<uint32_t>(__backends) & static_cast<uint32_t>(__target_backend)) != 0;
}

//! @brief Enumerates the different possibilities of data movement
//! @warning We do not allow inputs or outputs to have difference memory spaces. Either all inputs are on host or all
//! are on device.
enum class __memory_direction : uint8_t
{
  __host = 0, ///< input and output are on host
  __device, ///< input and output are on device
  __host_device, ///< input on host, output on device
  __device_host, ///< input on device, output on host
};

//! @brief Base class for our execution policies.
//! It takes an untagged uint32_t because we want to be able to store 3 different enumerations in it.
template <uint32_t _Policy>
struct __execution_policy_base
{
  template <uint32_t _OtherPolicy>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const __execution_policy_base&, const __execution_policy_base<_OtherPolicy>&) noexcept
  {
    return _Policy == _OtherPolicy;
  }

#if _CCCL_STD_VER <= 2017
  template <uint32_t _OtherPolicy>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const __execution_policy_base&, const __execution_policy_base<_OtherPolicy>&) noexcept
  {
    return _Policy != _OtherPolicy;
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Tag that identifies this and all derived classes as a CCCL execution policy
  static constexpr uint32_t __cccl_policy_ = _Policy;

  //! @brief Extracts the execution policy from the stored _Policy
  [[nodiscard]] _CCCL_API static constexpr __execution_policy __get_policy() noexcept
  {
    constexpr uint32_t __policy_mask{0x000000FF};
    return __execution_policy{_Policy & __policy_mask};
  }

  //! @brief Sets the execution policy
  //! @param __pol The new execution policy
  [[nodiscard]] _CCCL_API static constexpr uint32_t __set_policy(const __execution_policy __pol) noexcept
  {
    constexpr uint32_t __policy_mask{0xFFFFFF00};
    return (_Policy & __policy_mask) & static_cast<uint32_t>(__pol);
  }

  //! @brief Extracts the execution backend from the stored _Policy
  [[nodiscard]] _CCCL_API static constexpr __execution_backend __get_backend() noexcept
  {
    constexpr uint32_t __backend_mask{0x0000FF00};
    return __execution_backend{(_Policy & __backend_mask) >> 8};
  }

  //! @brief Sets the execution backend
  //! @param __pol The new backend
  [[nodiscard]] _CCCL_API static constexpr uint32_t __set_backend(const __execution_backend __pol) noexcept
  {
    constexpr uint32_t __backend_mask{0xFFFF00FF};
    return (_Policy & __backend_mask) & (static_cast<uint32_t>(__pol) << 8);
  }

  //! @brief Extracts the memory direction from the stored _Policy
  [[nodiscard]] _CCCL_API static constexpr __memory_direction __get_memory_direction() noexcept
  {
    constexpr uint32_t __direction_mask{0x00FF0000};
    return __memory_direction{(_Policy & __direction_mask) >> 16};
  }

  //! @brief Sets the memory direction
  //! @param __pol The new memory direction
  [[nodiscard]] _CCCL_API static constexpr uint32_t __set_backend(const __memory_direction __pol) noexcept
  {
    constexpr uint32_t __direction_mask{0xFF00FFFF};
    return (_Policy & __direction_mask) & (static_cast<uint32_t>(__pol) << 16);
  }

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
  [[nodiscard]] _CCCL_HOST_API static ::cuda::stream_ref get_stream() noexcept
  {
    return ::cuda::stream_ref{cudaStreamPerThread};
  }

  [[nodiscard]] _CCCL_HOST_API static auto set_stream(::cuda::stream_ref __stream) noexcept
  {
    return __execution_policy_stream<__execution_policy_base>{__stream};
  }
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
};

using sequenced_policy = __execution_policy_base<static_cast<uint32_t>(__execution_policy::__sequenced)>;
_CCCL_GLOBAL_CONSTANT sequenced_policy seq{};

using parallel_policy = __execution_policy_base<static_cast<uint32_t>(__execution_policy::__parallel)>;
_CCCL_GLOBAL_CONSTANT parallel_policy par{};

using parallel_unsequenced_policy =
  __execution_policy_base<static_cast<uint32_t>(__execution_policy::__parallel_unsequenced)>;
_CCCL_GLOBAL_CONSTANT parallel_unsequenced_policy par_unseq{};

using unsequenced_policy = __execution_policy_base<static_cast<uint32_t>(__execution_policy::__unsequenced)>;
_CCCL_GLOBAL_CONSTANT unsequenced_policy unseq{};

_CCCL_END_NAMESPACE_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXECUTION_POLICY_H
