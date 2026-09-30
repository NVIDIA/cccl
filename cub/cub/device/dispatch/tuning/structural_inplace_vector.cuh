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

// MSVC toolsets below 14.44 fail to correctly constant-fold structural_inplace_vector's bounds checks when this type
// is evaluated deep in CC dispatch's NTTP-based policy resolution, silently producing a zero-initialized element
// instead of hard-erroring at the actual out-of-bounds access. Disable the checks there; unaffected compilers (and
// newer MSVC) keep them.
#if !_CCCL_COMPILER(MSVC) || _CCCL_COMPILER(MSVC, >=, 19, 44)
#  define _CCCL_SIV_ASSERT(...) _CCCL_ASSERT(__VA_ARGS__)
#  define _CCCL_SIV_VERIFY(...) _CCCL_VERIFY(__VA_ARGS__)
#else
#  define _CCCL_SIV_ASSERT(...) ((void) 0)
#  define _CCCL_SIV_VERIFY(...) ((void) 0)
#endif

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

  // Kept as an aggregate (no user-declared constructors): older MSVC toolsets (< 19.44) fail to correctly
  // constant-fold this type through CC dispatch's NTTP-based policy resolution once it gains a user-declared
  // constructor (observed as a worker_policy silently reading back as zero-initialized deep in agent instantiation,
  // rather than a hard error at the actual fault). Callers list Capacity elements, using `{}` to pad unused slots,
  // and set `count` explicitly -- see e.g. make_baseline_policy().
  T elems[Capacity]{};
  size_type count = 0;

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
    _CCCL_SIV_ASSERT(pos < count, "structural_inplace_vector::operator[]: index out of range");
    return elems[pos];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_reference operator[](size_type pos) const noexcept
  {
    _CCCL_SIV_ASSERT(pos < count, "structural_inplace_vector::operator[]: index out of range");
    return elems[pos];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr reference at(size_type pos)
  {
    _CCCL_SIV_VERIFY(pos < count, "structural_inplace_vector::at: index out of range");
    return elems[pos];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_reference at(size_type pos) const
  {
    _CCCL_SIV_VERIFY(pos < count, "structural_inplace_vector::at: index out of range");
    return elems[pos];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr reference front() noexcept
  {
    _CCCL_SIV_ASSERT(count > 0, "structural_inplace_vector::front: empty vector");
    return elems[0];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_reference front() const noexcept
  {
    _CCCL_SIV_ASSERT(count > 0, "structural_inplace_vector::front: empty vector");
    return elems[0];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr reference back() noexcept
  {
    _CCCL_SIV_ASSERT(count > 0, "structural_inplace_vector::back: empty vector");
    return elems[count - 1];
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const_reference back() const noexcept
  {
    _CCCL_SIV_ASSERT(count > 0, "structural_inplace_vector::back: empty vector");
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

#undef _CCCL_SIV_ASSERT
#undef _CCCL_SIV_VERIFY
