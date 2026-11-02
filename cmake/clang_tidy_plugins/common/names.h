// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/StringRef.h>

namespace clang::tidy::cccl::util
{
[[nodiscard]] inline bool is_ugly_name(llvm::StringRef str)
{
  if (str.size() < 2)
  {
    return false;
  }

  if (str[0] == '_' && str[1] >= 'A' && str[1] <= 'Z')
  {
    // _Foo
    return true;
  }

  // __foo
  return str.starts_with("__");
}
} // namespace clang::tidy::cccl::util
