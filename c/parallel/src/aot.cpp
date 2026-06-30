//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cccl/c/aot.h>

extern "C" CCCL_C_API void cccl_aot_buffer_free(void* buf)
{
  // Buffers handed out by *_serialize are allocated with new[] in the
  // matching cccl_aot::buffer_writer::release implementation.
  delete[] static_cast<char*>(buf);
}
