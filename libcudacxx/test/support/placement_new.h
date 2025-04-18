//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PLACEMENT_NEW_HPP
#define PLACEMENT_NEW_HPP

// CUDA always defines placement new/delete for device code.
#if !defined(__CUDACC__)

#  include "test_macros.h"
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

#endif // !defined(__CUDACC__)

#endif // PLACEMENT_NEW_HPP
