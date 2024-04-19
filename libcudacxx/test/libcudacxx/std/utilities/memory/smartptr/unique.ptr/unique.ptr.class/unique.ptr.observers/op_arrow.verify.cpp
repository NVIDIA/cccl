//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// unique_ptr

// test op->()

#include <cuda/std/__memory_>

struct V
{
  int member;
};

void f()
{
  cuda::std::unique_ptr<V[]> p;
  cuda::std::unique_ptr<V[]> const& cp = p;

  p->member; // expected-error-re {{member reference type 'cuda::std::unique_ptr<V{{[ ]*}}[]>' is not a pointer}}
             // expected-error@-1 {{no member named 'member'}}

  cp->member; // expected-error-re {{member reference type 'const cuda::std::unique_ptr<V{{[ ]*}}[]>' is not a pointer}}
              // expected-error@-1 {{no member named 'member'}}
}
