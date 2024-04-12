//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: true

// optional

#include <cuda/std/iterator>
#include <cuda/std/optional>

static_assert(!cuda::std::indirectly_readable<cuda::std::optional<int>>);
static_assert(!cuda::std::indirectly_writable<cuda::std::optional<int>, int>);
static_assert(!cuda::std::weakly_incrementable<cuda::std::optional<int>>);
static_assert(!cuda::std::indirectly_movable<cuda::std::optional<int>, cuda::std::optional<int>>);
static_assert(!cuda::std::indirectly_movable_storable<cuda::std::optional<int>, cuda::std::optional<int>>);
static_assert(!cuda::std::indirectly_copyable<cuda::std::optional<int>, cuda::std::optional<int>>);
static_assert(!cuda::std::indirectly_copyable_storable<cuda::std::optional<int>, cuda::std::optional<int>>);
