//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "util/serialization.h"

#include <cstring>
#include <format>
#include <string>
#include <utility>

#include <cuda.h>

#include <cccl/c/serialization.h>
#include <cccl/c/serialization_diagnostics.h>
#include <cccl/c/types.h>

extern "C" CCCL_C_API void cccl_serialization_buffer_free(void* buf)
{
  // Buffers handed out by *_serialize are allocated with new[] in the
  // matching cccl::serialization::buffer_writer::release implementation.
  delete[] static_cast<char*>(buf);
}

namespace
{
// Per-thread last-error string. Set on a serialization failure, read via
// cccl_serialization_last_error(). Cleared at the start of cccl_serialization_validate_blob so a
// stale message from an earlier call is never reported for a later one.
thread_local std::string g_serialization_last_error;

void set_serialization_error(std::string msg)
{
  g_serialization_last_error = std::move(msg);
}

// Compute capability of the device the blob will most likely load on, or false
// if it cannot be determined. Prefers the current context's device; if there is
// no current context yet (a bare deserialize before any GPU work), falls back to
// the default device (0) — cuInit is idempotent and creates no context, so this
// has no side effects. On failure, cc validation is skipped and any
// incompatibility falls through to the driver at load time.
bool current_compute_capability(int& major, int& minor)
{
  CUdevice dev{};
  if (cuCtxGetDevice(&dev) != CUDA_SUCCESS)
  {
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS)
    {
      return false;
    }
  }
  if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS)
  {
    return false;
  }
  if (cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS)
  {
    return false;
  }
  return true;
}
} // namespace

extern "C" CCCL_C_API const char* cccl_serialization_last_error(void)
{
  return g_serialization_last_error.c_str();
}

extern "C" CCCL_C_API CUresult cccl_serialization_validate_blob(const void* buf, size_t size)
try
{
  using namespace cccl::serialization;
  g_serialization_last_error.clear();

  if (buf == nullptr || size < sizeof(blob_header))
  {
    set_serialization_error("serialization blob: buffer is null or smaller than the blob header");
    return CUDA_ERROR_INVALID_VALUE;
  }

  blob_header h{};
  std::memcpy(&h, buf, sizeof(h));

  if (std::memcmp(h.magic, k_blob_magic, sizeof(k_blob_magic)) != 0)
  {
    set_serialization_error("serialization blob: bad magic (not a CCCL serialization blob)");
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Reject an unrecognized payload_kind here too, mirroring read_and_validate_header
  // (used by every *_deserialize). Otherwise a corrupted payload_kind would pass this
  // pre-check and only fail later inside the deserialize call with a less-descriptive error.
  if (h.payload_kind != CCCL_PAYLOAD_LTOIR && h.payload_kind != CCCL_PAYLOAD_CUBIN)
  {
    set_serialization_error("serialization blob: unknown payload kind");
    return CUDA_ERROR_INVALID_VALUE;
  }

  // A CUBIN payload is final SASS, tied to the compute-capability major it was
  // built for; it is never binary-compatible across majors. Reject that up front
  // rather than failing deep inside cuLibraryLoadData with an opaque code. (A
  // differing minor is left to the driver, which is forward-compatible.)
  if (h.payload_kind == CCCL_PAYLOAD_CUBIN)
  {
    int major = 0;
    int minor = 0;
    if (current_compute_capability(major, minor))
    {
      const int blob_major = static_cast<int>(h.cc) / 10;
      if (blob_major != major)
      {
        set_serialization_error(std::format(
          "serialization blob targets sm_{} but the current device is sm_{}{}; a CUBIN payload is not compatible "
          "across "
          "compute-capability majors. Rebuild for this architecture (or ship one blob per target arch).",
          h.cc,
          major,
          minor));
        return CUDA_ERROR_NO_BINARY_FOR_GPU;
      }
    }
  }

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  set_serialization_error(std::string("serialization blob validation failed: ") + exc.what());
  return CUDA_ERROR_UNKNOWN;
}
