//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__DEVICE_ALL_DEVICES
#define _CUDAX__DEVICE_ALL_DEVICES

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/cassert>
#include <cuda/std/detail/libcxx/include/stdexcept>

#include <cuda/experimental/__device/device.cuh>

#include <vector>

namespace cuda::experimental
{
namespace detail
{
//! @brief A random-access range of all available CUDA devices
class all_devices
{
public:
  using size_type      = ::std::vector<device>::size_type;
  using iterator       = ::std::vector<device>::const_iterator;
  using const_iterator = ::std::vector<device>::const_iterator;

  all_devices() = default;

  _CCCL_NODISCARD const device& operator[](size_type __i) const noexcept;

  _CCCL_NODISCARD const device& at(size_type __i) const;

  _CCCL_NODISCARD size_type size() const;

  _CCCL_NODISCARD iterator begin() const noexcept;

  _CCCL_NODISCARD iterator end() const noexcept;

private:
  struct __initializer_iterator;

  static const ::std::vector<device>& __devices();
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

_CCCL_NODISCARD inline const device& all_devices::operator[](size_type __id_) const noexcept
{
  assert(__id_ < size());
  return __devices()[__id_];
}

_CCCL_NODISCARD inline const device& all_devices::at(size_type __id_) const
{
  if (__id_ >= size())
  {
    _CUDA_VSTD::__throw_out_of_range("device index out of range");
  }
  return __devices()[__id_];
}

_CCCL_NODISCARD inline all_devices::size_type all_devices::size() const
{
  return __devices().size();
}

_CCCL_NODISCARD inline all_devices::iterator all_devices::begin() const noexcept
{
  return __devices().begin();
}

_CCCL_NODISCARD inline all_devices::iterator all_devices::end() const noexcept
{
  return __devices().end();
}

inline const ::std::vector<device>& all_devices::__devices()
{
  static const ::std::vector<device> __devices = [] {
    int __count = 0;
    _CCCL_TRY_CUDA_API(::cudaGetDeviceCount, "failed to get the count of CUDA devices", &__count);
    return ::std::vector<device>{__initializer_iterator{0}, __initializer_iterator{__count}};
  }();
  return __devices;
}
} // namespace detail

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
//!   [[nodiscard]] constexpr const device& operator[](size_type i) const noexcept;
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
//! type of `const device&`.
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
inline constexpr detail::all_devices devices{};

inline const arch_traits_t& device_ref::arch_traits() const
{
  return devices[get()].arch_traits();
}

} // namespace cuda::experimental

#endif // _CUDAX__DEVICE_ALL_DEVICES
