// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <limits>

namespace c2h::detail
{
[[nodiscard]] inline long long parse_env_long_long(const char* value) noexcept
{
  if (value == nullptr)
  {
    return 0;
  }

  char* end = nullptr;
  errno     = 0;

  const long long result = std::strtoll(value, &end, 10);
  if (value == end || *end != '\0' || errno == ERANGE)
  {
    return 0;
  }

  return result;
}

[[nodiscard]] inline std::size_t parse_env_size(const char* value) noexcept
{
  if (value == nullptr)
  {
    return 0;
  }

  const char* first = value;
  while (std::isspace(static_cast<unsigned char>(*first)) != 0)
  {
    ++first;
  }
  if (*first == '-')
  {
    return 0;
  }

  char* end = nullptr;
  errno     = 0;

  const unsigned long long result = std::strtoull(value, &end, 10);
  if (value == end || *end != '\0' || errno == ERANGE)
  {
    return 0;
  }

  if constexpr (sizeof(unsigned long long) > sizeof(std::size_t))
  {
    if (result > static_cast<unsigned long long>((std::numeric_limits<std::size_t>::max)()))
    {
      return 0;
    }
  }

  return static_cast<std::size_t>(result);
}

[[nodiscard]] inline long long get_env_as_long_long(const char* name) noexcept
{
#ifdef _WIN32
  char* buf       = nullptr;
  std::size_t len = 0;
  if (_dupenv_s(&buf, &len, name) || !buf)
  {
    return 0;
  }
  const long long result = parse_env_long_long(buf);
  std::free(buf);
  return result;
#else
  if (const char* const v = std::getenv(name))
  {
    return parse_env_long_long(v);
  }
  return 0;
#endif
}

[[nodiscard]] inline std::size_t get_env_as_size(const char* name) noexcept
{
#ifdef _WIN32
  char* buf       = nullptr;
  std::size_t len = 0;
  if (_dupenv_s(&buf, &len, name) || !buf)
  {
    return 0;
  }
  const std::size_t result = parse_env_size(buf);
  std::free(buf);
  return result;
#else
  if (const char* const v = std::getenv(name))
  {
    return parse_env_size(v);
  }
  return 0;
#endif
}
} // namespace c2h::detail
