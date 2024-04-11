//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#define CHECK_MDSPAN_EXTENT(m, d, e0, e1) \
  static_assert(m.is_exhaustive(), "");   \
  assert(m.data_handle() == d.data());    \
  assert(m.rank() == 2);                  \
  assert(m.rank_dynamic() == 2);          \
  assert(m.extent(0) == e0);              \
  assert(m.extent(1) == e1)
