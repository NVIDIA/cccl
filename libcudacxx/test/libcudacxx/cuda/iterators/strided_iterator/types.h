//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CUDA_ITERATOR_STRIDED_ITERATOR_H
#define TEST_CUDA_ITERATOR_STRIDED_ITERATOR_H

#include <cuda/std/type_traits>

#include "test_macros.h"

template <cuda::std::ptrdiff_t Val = 2>
struct Stride : cuda::std::integral_constant<cuda::std::ptrdiff_t, Val>
{};
static_assert(::cuda::std::__integral_constant_like<Stride<2>>);

#endif // TEST_CUDA_ITERATOR_STRIDED_ITERATOR_H
