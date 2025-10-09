//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_COMPUTE_CAPABILITY_H
#define _CUDA___DEVICE_COMPUTE_CAPABILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/arch_id.h>
#include <cuda/__fwd/devices.h>
#include <cuda/std/__utility/to_underlying.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Type representing the CUDA compute capability.
class compute_capability
{
  int __cc_{}; //!< The stored compute capability in format 10 * major + minor.

public:
  _CCCL_HIDE_FROM_ABI constexpr compute_capability() noexcept = default;

  //! @brief Constructs the object from compute capability \c __cc. The expected format is 10 * major + minor.
  //!
  //! @param __cc Compute capability.
  _CCCL_API explicit constexpr compute_capability(int __cc) noexcept
      : __cc_{__cc}
  {}

  //! @brief Constructs the object by combining the \c __major and \c __minor compute capability.
  //!
  //! @param __major The major compute capability.
  //! @param __minor The minor compute capability. Must be less than 10.
  _CCCL_API constexpr compute_capability(int __major, int __minor) noexcept
      : __cc_{10 * __major + __minor}
  {
    _CCCL_ASSERT(__minor < 10, "invalid minor compute capability");
  }

  //! @brief Constructs the object from the architecture id.
  //!
  //! @param __arch_id The architecture id.
  _CCCL_API explicit constexpr compute_capability(arch_id __arch_id) noexcept
      : compute_capability{}
  {
    const auto __val = ::cuda::std::to_underlying(__arch_id);
    if (__val > __arch_specific_id_multiplier)
    {
      __cc_ = __val / __arch_specific_id_multiplier;
    }
    else
    {
      __cc_ = __val;
    }
  }

  _CCCL_HIDE_FROM_ABI constexpr compute_capability(const compute_capability&) noexcept = default;

  _CCCL_HIDE_FROM_ABI constexpr compute_capability& operator=(const compute_capability&) noexcept = default;

  //! @brief Gets the stored compute capability.
  //!
  //! @return The stored compute capability in format 10 * major + minor.
  [[nodiscard]] _CCCL_API constexpr int get() const noexcept
  {
    return __cc_;
  }

  //! @brief Gets the major compute capability.
  //!
  //! @return Major compute capability.
  [[nodiscard]] _CCCL_API constexpr int major() const noexcept
  {
    return __cc_ / 10;
  }

  //! @brief Gets the minor compute capability.
  //!
  //! @return Minor compute capability. The value is always less than 10.
  [[nodiscard]] _CCCL_API constexpr int minor() const noexcept
  {
    return __cc_ % 10;
  }

  //! @brief Conversion operator to \c int.
  //!
  //! @return The stored compute capability in format 10 * major + minor.
  _CCCL_API explicit constexpr operator int() const noexcept
  {
    return __cc_;
  }

  //! @brief Equality operator.
  [[nodiscard]] friend _CCCL_API constexpr bool operator==(compute_capability __lhs, compute_capability __rhs) noexcept
  {
    return __lhs.__cc_ == __rhs.__cc_;
  }

  //! @brief Inequality operator.
  [[nodiscard]] friend _CCCL_API constexpr bool operator!=(compute_capability __lhs, compute_capability __rhs) noexcept
  {
    return __lhs.__cc_ != __rhs.__cc_;
  }

  //! @brief Less than operator.
  [[nodiscard]] friend _CCCL_API constexpr bool operator<(compute_capability __lhs, compute_capability __rhs) noexcept
  {
    return __lhs.__cc_ < __rhs.__cc_;
  }

  //! @brief Less than or equal to operator.
  [[nodiscard]] friend _CCCL_API constexpr bool operator<=(compute_capability __lhs, compute_capability __rhs) noexcept
  {
    return __lhs.__cc_ <= __rhs.__cc_;
  }

  //! @brief Greater than operator.
  [[nodiscard]] friend _CCCL_API constexpr bool operator>(compute_capability __lhs, compute_capability __rhs) noexcept
  {
    return __lhs.__cc_ > __rhs.__cc_;
  }

  //! @brief Greater than or equal to operator.
  [[nodiscard]] friend _CCCL_API constexpr bool operator>=(compute_capability __lhs, compute_capability __rhs) noexcept
  {
    return __lhs.__cc_ >= __rhs.__cc_;
  }
};

inline constexpr int __lowest_known_arch  = 60;
inline constexpr int __highest_known_arch = 120;

[[nodiscard]] _CCCL_API inline constexpr bool __is_known_compute_capability(compute_capability __cc)
{
  return __cc.get() >= __lowest_known_arch && __cc.get() <= __highest_known_arch;
}

[[nodiscard]] _CCCL_API constexpr arch_id to_arch_id(compute_capability __cc) noexcept
{
  return static_cast<arch_id>(__cc.get());
}

[[nodiscard]] _CCCL_API constexpr arch_id to_arch_specific_id(compute_capability __cc) noexcept
{
  return static_cast<arch_id>(__cc.get() * __arch_specific_id_multiplier);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___DEVICE_COMPUTE_CAPABILITY_H
