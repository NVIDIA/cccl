// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string>

namespace c2h::detail
{
inline std::optional<std::string> get_env(const char* name)
{
#ifdef _WIN32
  char* buf       = nullptr;
  std::size_t len = 0;
  if (_dupenv_s(&buf, &len, name) || !buf)
  {
    return std::nullopt;
  }
  std::string val(buf);
  free(buf);
  return val;
#else
  if (const char* v = std::getenv(name))
  {
    return std::string(v);
  }
  return std::nullopt;
#endif
}
} // namespace c2h::detail
