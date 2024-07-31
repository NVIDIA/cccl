//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T, T N>
//   using make_integer_sequence = integer_sequence<T, 0, 1, ..., N-1>;

// This test hangs during recursive template instantiation with libstdc++
// UNSUPPORTED: stdlib=libstdc++

#define _LIBCUDACXX_TESTING_FALLBACK_MAKE_INTEGER_SEQUENCE
#include "make_integer_seq.fail.cpp"
