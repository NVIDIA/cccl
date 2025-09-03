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

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
#  include <cuda/__device/physical_device.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/cassert>
#  include <cuda/std/detail/libcxx/include/stdexcept>
#  include <cuda/std/span>

#  include <vector>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
namespace __detail
{
//! @brief A random-access range of all available CUDA devices
class all_devices
{
public:
  using size_type      = ::std::vector<physical_device>::size_type;
  using iterator       = ::std::vector<physical_device>::const_iterator;
  using const_iterator = ::std::vector<physical_device>::const_iterator;

  all_devices() = default;

  [[nodiscard]] const physical_device& operator[](size_type __i) const;

  [[nodiscard]] size_type size() const;

  [[nodiscard]] iterator begin() const noexcept;

  [[nodiscard]] iterator end() const noexcept;

  operator ::cuda::std::span<const device_ref>() const;

private:
  struct __initializer_iterator;

  static const ::std::vector<physical_device>& __devices();
};

//! @brief An iterator used to in-place construct `device` objects in a
//! std::vector.
//!
//! Since `device` objects are not movable or copyable, we need to construct them
//! in-place with a proxy object that can be implicitly converted to a `device`
//! object.
struct all_devices::__initializer_iterator
{
  using value_type        = __emplace_device;
  using reference         = __emplace_device;
  using iterator_category = ::std::forward_iterator_tag;
  using difference_type   = int;
  using pointer           = __emplace_device;

  int __id_;

  __emplace_device operator*() const noexcept
  {
    return __emplace_device{__id_};
  }

  __emplace_device operator->() const noexcept
  {
    return __emplace_device{__id_};
  }

  __initializer_iterator& operator++() noexcept
  {
    ++__id_;
    return *this;
  }

  __initializer_iterator operator++(int) noexcept
  {
    auto __tmp = *this;
    ++__id_;
    return __tmp;
  }

  bool operator==(const __initializer_iterator& __other) const noexcept
  {
    return __id_ == __other.__id_;
  }

  bool operator!=(const __initializer_iterator& __other) const noexcept
  {
    return __id_ != __other.__id_;
  }
};

[[nodiscard]] inline const physical_device& all_devices::operator[](size_type __id_) const
{
  if (__id_ >= size())
  {
    if (size() == 0)
    {
      ::cuda::std::__throw_out_of_range("device was requested but no CUDA devices found");
    }
    else
    {
      ::cuda::std::__throw_out_of_range(
        (::std::string("device index out of range: ") + ::std::to_string(__id_)).c_str());
    }
  }
  return __devices()[__id_];
}

[[nodiscard]] inline all_devices::size_type all_devices::size() const
{
  return __devices().size();
}

[[nodiscard]] inline all_devices::iterator all_devices::begin() const noexcept
{
  return __devices().begin();
}

[[nodiscard]] inline all_devices::iterator all_devices::end() const noexcept
{
  return __devices().end();
}

inline all_devices::operator ::cuda::std::span<const device_ref>() const
{
  static const ::std::vector<device_ref> __refs(begin(), end());
  return ::cuda::std::span<const device_ref>(__refs);
}

inline const ::std::vector<physical_device>& all_devices::__devices()
{
  static const ::std::vector<physical_device> __devices = [] {
    int __count = 0;
    _CCCL_TRY_CUDA_API(::cudaGetDeviceCount, "failed to get the count of CUDA devices", &__count);
    return ::std::vector<physical_device>{__initializer_iterator{0}, __initializer_iterator{__count}};
  }();
  return __devices;
}
} // namespace __detail

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
//!   [[nodiscard]] constexpr const physical_device& operator[](size_type i) const noexcept;
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
//! type of `const physical_device&`.
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
inline constexpr __detail::all_devices devices{};

inline const arch::traits_t& device_ref::arch_traits() const
{
  return devices[get()].arch_traits();
}

[[nodiscard]] inline ::std::vector<device_ref> device_ref::peer_devices() const
{
  ::std::vector<device_ref> __result;
  __result.reserve(devices.size());

  for (const physical_device& __other_dev : devices)
  {
    // Exclude the device this API is called on. The main use case for this API
    // is enable/disable peer access. While enable peer access can be called on
    // device on which memory resides, disable peer access will error-out.
    // Usage of the peer access control is smoother when *this is excluded,
    // while it can be easily added with .push_back() on the vector if a full
    // group of peers is needed (for cases other than peer access control)
    if (__other_dev != *this)
    {
      // While in almost all practical applications peer access should be symmetrical,
      // it is possible to build a system with one directional peer access, check
      // both ways here just to be safe
      if (has_peer_access_to(__other_dev) && __other_dev.has_peer_access_to(*this))
      {
        __result.push_back(__other_dev);
      }
    }
  }
  return __result;
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DEVICE_ALL_DEVICES_H
