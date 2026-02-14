//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_ALL_DEVICES_H
#define _CUDA___DEVICE_ALL_DEVICES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__device/device_ref.h>
#  include <cuda/__device/physical_device.h>
#  include <cuda/__driver/driver_api.h>
#  include <cuda/__fwd/devices.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/span>

#  include <vector>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

[[nodiscard]] _CCCL_HOST_API inline ::std::vector<device_ref> __make_devices()
{
  ::std::vector<device_ref> __ret{};
  __ret.reserve(::cuda::__physical_devices().size());
  for (::cuda::std::size_t __i = 0; __i < ::cuda::__physical_devices().size(); ++__i)
  {
    __ret.emplace_back(static_cast<int>(__i));
  }
  return __ret;
}

[[nodiscard]] inline ::cuda::std::span<const device_ref> __devices()
{
  static const auto __devices = ::cuda::__make_devices();
  return ::cuda::std::span<const device_ref>{__devices.data(), __devices.size()};
}

//! @brief A random-access range of all available CUDA devices
class __all_devices
{
public:
  using value_type = ::cuda::std::span<const device_ref>::value_type;
  using size_type  = ::cuda::std::span<const device_ref>::size_type;
  using iterator   = ::cuda::std::span<const device_ref>::iterator;

  _CCCL_HIDE_FROM_ABI __all_devices()            = default;
  __all_devices(const __all_devices&)            = delete;
  __all_devices(__all_devices&&)                 = delete;
  __all_devices& operator=(const __all_devices&) = delete;
  __all_devices& operator=(__all_devices&&)      = delete;

  [[nodiscard]] _CCCL_HOST_API device_ref operator[](size_type __i) const
  {
    if (__i >= size())
    {
      ::cuda::std::__throw_out_of_range("device index out of range");
    }
    return ::cuda::__devices()[__i];
  }

  [[nodiscard]] _CCCL_HOST_API size_type size() const
  {
    return ::cuda::__devices().size();
  }

  [[nodiscard]] _CCCL_HOST_API iterator begin() const
  {
    return ::cuda::__devices().begin();
  }

  [[nodiscard]] _CCCL_HOST_API iterator end() const
  {
    return ::cuda::__devices().end();
  }
};

//! @brief A range of all available CUDA devices
//!
//! `cuda::devices` provides a view of all available CUDA devices. It is useful for
//! determining the number of supported devices and for iterating over all devices
//! in a range-based for loop (e.g., to print device properties, perhaps).
//!
//! @par Class synopsis
//! @code
//! class __all_devices {                     // exposition only
//! public:
//!   using size_type = ::std::size_t;
//!   struct iterator;
//!   using const_iterator = iterator;
//!
//!   [[nodiscard]] device_ref operator[](size_type i) const noexcept;
//!
//!   [[nodiscard]] size_type size() const;
//!
//!   [[nodiscard]] iterator begin() const noexcept;
//!
//!   [[nodiscard]] iterator end() const noexcept;
//! };
//! @endcode
//!
//! @par
//! `__all_devices::iterator` is a random access iterator with a `reference`
//! type of `const device_ref&`.
//!
//! @par Example
//! @code
//! auto& dev0 = cuda::devices[0];
//! assert(cuda::devices.size() == cuda::std::distance(cuda::devices.begin(), cuda::devices.end()));
//! @endcode
//!
//! @sa
//! * device
//! * device_ref
inline constexpr __all_devices devices{};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DEVICE_ALL_DEVICES_H
