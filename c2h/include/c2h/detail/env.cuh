// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/std/charconv>
#include <cuda/std/type_traits>

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace c2h::detail
{
template <typename T>
[[nodiscard]] inline T parse_env_integer(const char* value) noexcept
{
  static_assert(::cuda::std::is_integral_v<T>);

  if (value == nullptr)
  {
    return T{};
  }

  const char* const end = value + ::std::strlen(value);
  T result{};
  const auto conversion_result = ::cuda::std::from_chars(value, end, result);
  if (conversion_result.ec != ::cuda::std::errc{} || conversion_result.ptr != end)
  {
    return T{};
  }

  return result;
}

template <typename T>
[[nodiscard]] inline T get_env_as_integer(const char* name) noexcept
{
#ifdef _WIN32
  char* buf         = nullptr;
  ::std::size_t len = 0;
  if (_dupenv_s(&buf, &len, name) || !buf)
  {
    return T{};
  }
  const T result = parse_env_integer<T>(buf);
  ::std::free(buf);
  return result;
#else
  return parse_env_integer<T>(::std::getenv(name));
#endif
}
} // namespace c2h::detail
