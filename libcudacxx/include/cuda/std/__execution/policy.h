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
  __none = 0,
#if _CCCL_HAS_BACKEND_CUDA()
  __cuda = 1 << 1,
#endif // _CCCL_HAS_BACKEND_CUDA()
#if _CCCL_HAS_BACKEND_OMP()
  __omp = 1 << 2,
#endif // _CCCL_HAS_BACKEND_OMP()
#if _CCCL_HAS_BACKEND_TBB()
  __tbb = 1 << 3,
#endif // _CCCL_HAS_BACKEND_TBB()
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

  //! @brief Extracts the execution backend from the stored _Policy
  [[nodiscard]] _CCCL_API static constexpr __execution_backend __get_backend() noexcept
  {
    constexpr uint32_t __backend_mask{0x0000FF00};
    return __execution_backend{(_Policy & __backend_mask) >> 8};
  }
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
