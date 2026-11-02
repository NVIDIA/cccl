// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string_view>

#include <c2h/allocation_failure.h>
#include <catch2/catch_test_macros.hpp>

namespace c2h
{
// Lippincott handler: call from catch (...) to skip allocation failures and
// rethrow other exceptions. This must allow Catch2's skip exception to escape.
inline void skip_if_allocation_failure(std::string_view message = "Skipping test: insufficient memory")
{
  rethrow_unless_allocation_failure();
  CATCH_SKIP(message);
}
} // namespace c2h
