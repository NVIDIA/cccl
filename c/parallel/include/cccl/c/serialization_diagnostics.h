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

#include <cuda.h>
#include <stddef.h>

#include <cccl/c/extern_c.h>
#include <cccl/c/types.h>

// Diagnostics for the deserialize path. Declared in a standalone header
// (rather than serialization.h, which is pulled in transitively by every algorithm
// translation unit) so adding it does not trigger a rebuild of the whole C
// parallel library.

CCCL_C_EXTERN_C_BEGIN

// Returns a human-readable description of the most recent serialization failure on the
// calling thread, or "" if none. The returned pointer is owned by the library
// and is valid until the next serialization call on the same thread. Callers can surface
// this so a deserialize failure carries an actionable message instead of only an
// opaque CUDA error code.
CCCL_C_API const char* cccl_serialization_last_error(void);

// Validates that a serialization blob (a build_result blob, i.e. what is passed to
// cccl_device_<algo>_deserialize) can be loaded on the current device, *before*
// the opaque cuLibraryLoadData failure. Checks the blob magic, format version,
// CCCL C parallel ABI version, and — for CUBIN payloads — that the target
// compute-capability major matches the current device.
//
// Returns CUDA_SUCCESS if the blob looks loadable, or an error code with a
// message retrievable via cccl_serialization_last_error(). This never executes device
// code; it only inspects the header.
CCCL_C_API CUresult cccl_serialization_validate_blob(const void* buf, size_t size);

CCCL_C_EXTERN_C_END
// NOLINTEND(modernize-use-using)
