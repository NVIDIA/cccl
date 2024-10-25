//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/utility/hash.cuh>

#include <atomic>

namespace cuda::experimental::stf
{

namespace reserved
{

/* This defines an object with a unique identifier. This object is non
 * copyable, but moving it transfers the unique id to the destination object.
 */
template <typename C, C...>
class unique_id
{
public:
  unique_id() = default;
  constexpr unique_id(unique_id&& other) noexcept
      : _value(::std::exchange(other._value, -1))
  {}
  constexpr unique_id(const int val) noexcept
      : _value(val)
  {}
  constexpr operator int() const noexcept
  {
    return _value;
  }

  unique_id& operator=(unique_id&& rhs)
  {
    _value = ::std::exchange(rhs._value, -1);
    return *this;
  }

  unique_id(const unique_id&)            = delete;
  unique_id& operator=(const unique_id&) = delete;

  bool operator==(const unique_id& other) const noexcept
  {
    return _value == other._value;
  }

private:
  static int next_id()
  {
    static ::std::atomic<int> id = 0;
    return id++;
  }

  int _value = next_id();
};

} // end namespace reserved

template <typename C, C... letters>
struct hash<reserved::unique_id<C, letters...>>
{
  size_t operator()(const reserved::unique_id<C, letters...>& id) const
  {
    return ::std::hash<int>()(id);
  }
};

} // end namespace cuda::experimental::stf
