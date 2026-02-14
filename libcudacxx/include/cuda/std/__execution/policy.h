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

#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__fwd/execution_policy.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

[[nodiscard]] _CCCL_API constexpr bool __has_unique_backend(const __execution_backend __backends) noexcept
{
  return ::cuda::std::has_single_bit(static_cast<uint32_t>(__backends));
}

//! @brief Base class for our execution policies.
//! It takes an untagged uint32_t because we want to be able to store 3 different enumerations in it.
template <uint32_t _Policy, __execution_backend _Backend>
struct __execution_policy_base
{
  //! @brief Tag that identifies this and all derived classes as a CCCL execution policy
  static constexpr uint32_t __cccl_policy_ = _Policy;

  template <uint32_t _OtherPolicy, __execution_backend _OtherBackend>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const __execution_policy_base&, const __execution_policy_base<_OtherPolicy, _OtherBackend>&) noexcept
  {
    return _Policy == _OtherPolicy;
  }

#if _CCCL_STD_VER <= 2017
  template <uint32_t _OtherPolicy, __execution_backend _OtherBackend>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const __execution_policy_base&, const __execution_policy_base<_OtherPolicy, _OtherBackend>&) noexcept
  {
    return _Policy != _OtherPolicy;
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Extracts the execution policy from the stored _Policy
  [[nodiscard]] _CCCL_API static constexpr __execution_policy __get_policy() noexcept
  {
    return __policy_to_execution_policy<_Policy>;
  }

  //! @brief Extracts the execution backend from the stored _Policy
  [[nodiscard]] _CCCL_API static constexpr __execution_backend __get_backend() noexcept
  {
    return __policy_to_execution_backend<_Policy>;
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

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXECUTION_POLICY_H
