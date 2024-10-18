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

#include <atomic>

namespace cuda::experimental::stf
{

/* This defines an object with a unique identifier. This object is non
 * copyable, but moving it transfers the unique id to the destination object.
 */
template <typename C, C...>
class unique_numeral
{
public:
  unique_numeral() = default;
  constexpr unique_numeral(unique_numeral&& other) noexcept
      : _value(::std::exchange(other._value, -1))
  {}
  constexpr unique_numeral(const int val) noexcept
      : _value(val)
  {}
  constexpr operator int() const noexcept
  {
    return _value;
  }

  unique_numeral& operator=(unique_numeral&& rhs)
  {
    _value = ::std::exchange(rhs._value, -1);
    return *this;
  }

  unique_numeral(const unique_numeral&)            = delete;
  unique_numeral& operator=(const unique_numeral&) = delete;

  bool operator==(const unique_numeral& other) const noexcept
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

template <typename C, C... letters>
auto operator""_unique_id()
{
  return unique_numeral<C, letters...>();
}

} // end namespace cuda::experimental::stf

template <typename C, C... letters>
struct std::hash<cuda::experimental::stf::unique_numeral<C, letters...>>
{
  size_t operator()(const cuda::experimental::stf::unique_numeral<C, letters...>& id) const
  {
    return ::std::hash<int>()(id);
  }
};
