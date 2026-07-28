//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// enum class range_format {
//   disabled,
//   map,
//   set,
//   sequence,
//   string,
//   debug_string
// };

#include <cuda/std/__format_>

// test that the enumeration values exist
static_assert(cuda::std::range_format::disabled == cuda::std::range_format::disabled);
static_assert(cuda::std::range_format::map == cuda::std::range_format::map);
static_assert(cuda::std::range_format::set == cuda::std::range_format::set);
static_assert(cuda::std::range_format::sequence == cuda::std::range_format::sequence);
static_assert(cuda::std::range_format::string == cuda::std::range_format::string);
static_assert(cuda::std::range_format::debug_string == cuda::std::range_format::debug_string);

int main(int, char**)
{
  return 0;
}
