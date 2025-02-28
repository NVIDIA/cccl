//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// <mdspan>

// template<class Extents>
// class layout_left::mapping;

// If Extents is not a specialization of extents, then the program is
// ill-formed.

// Mandates: If Extents::rank_dynamic() == 0 is true, then the size of the
// multidimensional index space Extents() is representable as a value of type
// typename Extents::index_type.

#include <cuda/std/mdspan>

#include "test_macros.h"

__host__ __device__ void not_extents()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}layout_right::mapping template argument
  // must be a specialization of extents}}
  cuda::std::layout_right::mapping<void> mapping;
  unused(mapping);
}

__host__ __device__ void representable()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}layout_right::mapping product of static
  // extents must be representable as index_type.}}
  cuda::std::layout_right::mapping<cuda::std::extents<char, 20, 20>> mapping;
  unused(mapping)
}

int main(int, char**)
{
  return 0;
}
