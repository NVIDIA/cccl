//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once
// NOLINTBEGIN(modernize-use-using)

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#include <stddef.h>
#include <stdint.h>

#include <cccl/c/extern_c.h>
#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

// Algorithm tag stored in the blob header. Used to detect cross-algorithm
// deserialization attempts (e.g. loading a reduce blob with scan_deserialize).
typedef enum cccl_aot_algo_t
{
  CCCL_AOT_ALGO_REDUCE              = 1,
  CCCL_AOT_ALGO_SCAN                = 2,
  CCCL_AOT_ALGO_SEGMENTED_REDUCE    = 3,
  CCCL_AOT_ALGO_TRANSFORM           = 4,
  CCCL_AOT_ALGO_BINARY_SEARCH       = 5,
  CCCL_AOT_ALGO_MERGE_SORT          = 6,
  CCCL_AOT_ALGO_RADIX_SORT          = 7,
  CCCL_AOT_ALGO_SEGMENTED_SORT      = 8,
  CCCL_AOT_ALGO_THREE_WAY_PARTITION = 9,
  CCCL_AOT_ALGO_UNIQUE_BY_KEY       = 10,
  CCCL_AOT_ALGO_HISTOGRAM           = 11,
  CCCL_AOT_ALGO_FOR                 = 12,
} cccl_aot_algo_t;

// Version of the CCCL C parallel AoT blob format. Increment whenever a
// breaking change is made to the C parallel layer (build_result layouts,
// serialization wire format, etc.) independently of CCCL_VERSION.
#define CCCL_C_PARALLEL_VERSION 1

// Frees a buffer returned by any cccl_device_<algo>_serialize call.
// Required because allocations cross the cccl.c.parallel shared-library
// boundary; callers must not free returned buffers themselves.
CCCL_C_API void cccl_aot_buffer_free(void* buf);

CCCL_C_EXTERN_C_END
// NOLINTEND(modernize-use-using)
