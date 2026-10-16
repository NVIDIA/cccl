//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
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
// Deliberately a SEPARATE enum from v1's cccl_serialization_algo_t (c/parallel,
// NVRTC backend) — v2 blobs are distinguished from v1 blobs by magic
// (CCCLSEV2 vs CCCLSER1) and are never interchangeable, so there is no need
// for the numeric values to match.
typedef enum cccl_serialization_algo_v2_t
{
  CCCL_SERIALIZATION_V2_ALGO_REDUCE              = 0,
  CCCL_SERIALIZATION_V2_ALGO_SCAN                = 1,
  CCCL_SERIALIZATION_V2_ALGO_SEGMENTED_REDUCE    = 2,
  CCCL_SERIALIZATION_V2_ALGO_UNARY_TRANSFORM     = 3,
  CCCL_SERIALIZATION_V2_ALGO_BINARY_TRANSFORM    = 4,
  CCCL_SERIALIZATION_V2_ALGO_BINARY_SEARCH       = 5,
  CCCL_SERIALIZATION_V2_ALGO_MERGE_SORT          = 6,
  CCCL_SERIALIZATION_V2_ALGO_RADIX_SORT          = 7,
  CCCL_SERIALIZATION_V2_ALGO_SEGMENTED_SORT      = 8,
  CCCL_SERIALIZATION_V2_ALGO_THREE_WAY_PARTITION = 9,
  CCCL_SERIALIZATION_V2_ALGO_UNIQUE_BY_KEY       = 10,
  CCCL_SERIALIZATION_V2_ALGO_HISTOGRAM           = 11,
} cccl_serialization_algo_v2_t;

// Frees a buffer returned by any cccl_device_<algo>_serialize call.
// Required because allocations cross the cccl.c.parallel.v2 shared-library
// boundary; callers must not free returned buffers themselves.
CCCL_C_API void cccl_serialization_v2_buffer_free(void* buf);

CCCL_C_EXTERN_C_END
// NOLINTEND(modernize-use-using)
