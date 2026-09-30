// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/assert.h>
#include <cuda/std/cstddef>
#include <cuda/std/initializer_list>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! Like inplace_vector<T, N>, but structural, so we can pass it as part of NTTPs (e.g. tuning policies)
template <typename T, ::cuda::std::size_t Capacity>
struct structural_inplace_vector
{
  using value_type      = T;
  using size_type       = ::cuda::std::size_t;
  using difference_type = ::cuda::std::ptrdiff_t;
  using reference       = T&;
  using const_reference = const T&;
  using pointer         = T*;
  using const_pointer   = const T*;
  using iterator        = T*;
  using const_iterator  = const T*;

  T elems[Capacity]{};
  size_type count = 0;

  constexpr structural_inplace_vector() = default;

  _CCCL_HOST_DEVICE_API constexpr structural_inplace_vector(::cuda::std::initializer_list<T> ilist)
  {
    _CCCL_ASSERT(ilist.size() <= Capacity, "structural_inplace_vector: initializer list exceeds capacity");
    for (const auto& elem : ilist)
    {
      elems[count++] = elem;
    }
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool empty() const noexcept
  {
    return count == 0;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr size_type size() const noexcept
  {
    return count;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr size_type max_size() const noexcept
  {
    return Capacity;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr reference operator[](size_type pos) noexcept
  {
    _CCCL_ASSERT(pos < count, "structural_inplace_vector::operator[]: index out of range");
    return elems[pos];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_reference operator[](size_type pos) const noexcept
  {
    _CCCL_ASSERT(pos < count, "structural_inplace_vector::operator[]: index out of range");
    return elems[pos];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr reference at(size_type pos)
  {
    _CCCL_VERIFY(pos < count, "structural_inplace_vector::at: index out of range");
    return elems[pos];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_reference at(size_type pos) const
  {
    _CCCL_VERIFY(pos < count, "structural_inplace_vector::at: index out of range");
    return elems[pos];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr reference front() noexcept
  {
    _CCCL_ASSERT(count > 0, "structural_inplace_vector::front: empty vector");
    return elems[0];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_reference front() const noexcept
  {
    _CCCL_ASSERT(count > 0, "structural_inplace_vector::front: empty vector");
    return elems[0];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr reference back() noexcept
  {
    _CCCL_ASSERT(count > 0, "structural_inplace_vector::back: empty vector");
    return elems[count - 1];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_reference back() const noexcept
  {
    _CCCL_ASSERT(count > 0, "structural_inplace_vector::back: empty vector");
    return elems[count - 1];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr pointer data() noexcept
  {
    return elems;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_pointer data() const noexcept
  {
    return elems;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr iterator begin() noexcept
  {
    return elems;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_iterator begin() const noexcept
  {
    return elems;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr iterator end() noexcept
  {
    return elems + count;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_iterator end() const noexcept
  {
    return elems + count;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const structural_inplace_vector& lhs, const structural_inplace_vector& rhs)
  {
    if (lhs.count != rhs.count)
    {
      return false;
    }
    for (size_type i = 0; i < lhs.count; ++i)
    {
      if (lhs.elems[i] != rhs.elems[i])
      {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const structural_inplace_vector& lhs, const structural_inplace_vector& rhs)
  {
    return !(lhs == rhs);
  }
};
} // namespace detail

CUB_NAMESPACE_END
