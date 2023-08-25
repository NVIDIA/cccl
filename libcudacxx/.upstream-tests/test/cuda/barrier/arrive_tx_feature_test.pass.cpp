//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-90

// <cuda/barrier>

#include <cuda/barrier>

#ifndef __cccl_lib_has_experimental_ctk12_cp_async_exposure
static_assert(false, "should define __cccl_lib_has_experimental_ctk12_cp_async_exposure for SM_90 up");
#endif // __cccl_lib_has_experimental_ctk12_cp_async_exposure

int main(int, char**)
{}
