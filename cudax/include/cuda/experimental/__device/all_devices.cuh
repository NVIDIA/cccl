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

#include <cuda/std/__cccl/attributes.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/atomic>
#include <cuda/std/cassert> // IWYU pragma: keep (for assert)
#include <cuda/std/iterator> // IWYU pragma: keep (for random_access_iterator_tag)

#include <cuda/experimental/__device/device.cuh>

namespace cuda::experimental
{
namespace __detail
{
//! @brief A random-access range of all available CUDA devices
class all_devices
{
public:
  using size_type = _CUDA_VSTD::size_t;
  struct iterator;
  using const_iterator = iterator;

  all_devices() = default;

  _CCCL_NODISCARD constexpr device operator[](size_type __i) const noexcept;

  _CCCL_NODISCARD size_type size() const;

  _CCCL_NODISCARD iterator begin() const noexcept;

  _CCCL_NODISCARD iterator end() const noexcept;

private:
  // The number of available devices is not expected to change during the
  // lifetime of the program, so cache the count to avoid repeated calls to
  // `cudaGetDeviceCount`.
  mutable _CUDA_VSTD::atomic<size_type> __size_{~0UL};
};

//! @brief A random-access iterator over all available CUDA devices
struct all_devices::iterator
{
  using value_type        = device;
  using difference_type   = int;
  using pointer           = const device*;
  using reference         = device;
  using iterator_category = ::cuda::std::random_access_iterator_tag;
  using iterator_concept  = ::cuda::std::random_access_iterator_tag;

  //! @brief Proxy object used as the return value of `operator->`
  struct __proxy
  {
    _CCCL_NODISCARD const device* operator->() const noexcept
    {
      return &__dev_;
    }

    device __dev_;
  };

  _CCCL_NODISCARD constexpr device operator*() const noexcept
  {
    return device(__id_);
  }

  _CCCL_NODISCARD constexpr device operator[](int __offset) const noexcept
  {
    return device(__id_ + __offset);
  }

  _CCCL_NODISCARD constexpr __proxy operator->() const noexcept
  {
    return __proxy{device(__id_)};
  }

  constexpr iterator& operator++() noexcept
  {
    ++__id_;
    return *this;
  }

  constexpr iterator operator++(int) noexcept
  {
    auto tmp = *this;
    ++__id_;
    return tmp;
  }

  constexpr iterator& operator--() noexcept
  {
    --__id_;
    return *this;
  }

  constexpr iterator operator--(int) noexcept
  {
    auto tmp = *this;
    --__id_;
    return tmp;
  }

  constexpr iterator& operator+=(int __offset) noexcept
  {
    __id_ += __offset;
    return *this;
  }

  _CCCL_NODISCARD constexpr iterator operator+(int __offset) noexcept
  {
    return iterator(__id_ + __offset);
  }

  constexpr iterator& operator-=(int __offset) noexcept
  {
    __id_ -= __offset;
    return *this;
  }

  _CCCL_NODISCARD constexpr iterator operator-(int __offset) noexcept
  {
    return iterator(__id_ - __offset);
  }

  _CCCL_NODISCARD_FRIEND constexpr iterator operator+(int __lhs, const iterator& __rhs) noexcept
  {
    return iterator(__lhs + __rhs.__id_);
  }

  _CCCL_NODISCARD_FRIEND constexpr int operator-(const iterator& __lhs, const iterator& __rhs) noexcept
  {
    return __lhs.__id_ - __rhs.__id_;
  }

  _CCCL_NODISCARD_FRIEND constexpr bool operator==(const iterator& __lhs, const iterator& __rhs) noexcept
  {
    return __lhs.__id_ == __rhs.__id_;
  }

  _CCCL_NODISCARD_FRIEND constexpr bool operator!=(const iterator& __lhs, const iterator& __rhs) noexcept
  {
    return __lhs.__id_ != __rhs.__id_;
  }

private:
  friend struct all_devices;

  constexpr explicit iterator(int __id) noexcept
      : __id_(__id)
  {}

  int __id_ = 0;
};

_CCCL_NODISCARD inline constexpr device all_devices::operator[](size_type __id_) const noexcept
{
  assert(__id_ < size());
  return device(static_cast<int>(__id_));
}

_CCCL_NODISCARD inline all_devices::size_type all_devices::size() const
{
  size_type __size = __size_.load(_CUDA_VSTD::memory_order_relaxed);
  if (__size == ~0UL)
  {
    int __count = 0;
    _CCCL_TRY_CUDA_API(::cudaGetDeviceCount, "failed to get the count of CUDA devices", &__count);
    __size = static_cast<size_type>(__count);
    __size_.store(__size, _CUDA_VSTD::memory_order_relaxed);
  }
  return __size;
}

_CCCL_NODISCARD inline all_devices::iterator all_devices::begin() const noexcept
{
  return iterator(0);
}

_CCCL_NODISCARD inline all_devices::iterator all_devices::end() const noexcept
{
  return iterator(static_cast<int>(size()));
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
//!   using size_type = ::cuda::std::size_t;
//!   struct iterator;
//!   using const_iterator = iterator;
//!
//!   [[nodiscard]] constexpr device operator[](size_type i) const noexcept;
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
//! type of `device`.
//!
//! @par Example
//! @code
//! auto dev0 = cuda::devices[0];
//! assert(cuda::devices.size() == cuda::std::distance(cuda::devices.begin(), cuda::devices.end()));
//! @endcode
//!
//! @sa device
inline constexpr __detail::all_devices devices{};

} // namespace cuda::experimental

#endif // _CUDAX__DEVICE_ALL_DEVICES
