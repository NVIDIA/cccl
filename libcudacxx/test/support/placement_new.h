//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PLACEMENT_NEW_HPP
#define PLACEMENT_NEW_HPP

#include "test_macros.h"

// CUDA always defines placement new/delete for device code.
#if !_CCCL_CUDA_COMPILATION()

#  include <stddef.h> // Avoid depending on the C++ standard library.

void* operator new(size_t, void* p)
{
  return p;
}
void* operator new[](size_t, void* p)
{
  return p;
}
void operator delete(void*, void*) {}
void operator delete[](void*, void*) {}

#endif // !_CCCL_CUDA_COMPILATION()

#endif // PLACEMENT_NEW_HPP
