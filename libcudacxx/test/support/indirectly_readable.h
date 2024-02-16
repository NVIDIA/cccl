// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef LIBCXX_TEST_SUPPORT_INDIRECTLY_READABLE_H
#define LIBCXX_TEST_SUPPORT_INDIRECTLY_READABLE_H

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class Token>
struct Common { };

template <class Token>
struct T1 : Common<Token> { };

template <class Token>
struct T2 : Common<Token> { };

template <template <class> class T1Qual, template <class> class T2Qual, class Token>
struct cuda::std::basic_common_reference<T1<Token>, T2<Token>, T1Qual, T2Qual> {
  using type = Common<Token>;
};
template <template <class> class T2Qual, template <class> class T1Qual, class Token>
struct cuda::std::basic_common_reference<T2<Token>, T1<Token>, T2Qual, T1Qual>
  : cuda::std::basic_common_reference<T1<Token>, T2<Token>, T1Qual, T2Qual>
{ };

template <class Token>
struct IndirectlyReadable {
  using value_type = T1<Token>;
  TEST_HOST_DEVICE T2<Token>& operator*() const;
};

#endif // LIBCXX_TEST_SUPPORT_INDIRECTLY_READABLE_H
