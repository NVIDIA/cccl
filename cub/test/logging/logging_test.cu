// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This test is driven from the CMakeLists.txt in the same directory

#include <cub/detail/logging.cuh>

#include <iostream>
#include <string>

// Sentinels grepped for by ctest. Keep in sync with test/logging/CMakeLists.txt.
static constexpr auto log_sentinel        = "CCCL_LOG_SENTINEL";
static constexpr auto log_always_sentinel = "CCCL_LOG_ALWAYS_SENTINEL";

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cerr << "logging_test: expected exactly one of --expect-enabled or --expect-disabled\n";
    return 2;
  }

  const std::string arg = argv[1];
  bool expect_enabled;
  if (arg == "--expect-enabled")
  {
    expect_enabled = true;
  }
  else if (arg == "--expect-disabled")
  {
    expect_enabled = false;
  }
  else
  {
    std::cerr << "logging_test: unknown argument '" << arg << "'\n";
    return 2;
  }

  const bool enabled = cub::detail::logging_enabled();
  if (enabled != expect_enabled)
  {
    std::cerr
      << "logging_test: logging_enabled() == " << std::boolalpha << enabled << ", expected " << expect_enabled << '\n';
    return 1;
  }

  // Emits its sentinel only when logging is enabled (and compiled in).
  cub::detail::log("%s\n", log_sentinel);
  // Emits its sentinel whenever logging is compiled in, independently of the environment.
  cub::detail::log_always("%s\n", log_always_sentinel);
  return 0;
}
