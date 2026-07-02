//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <format>
#include <string>
#include <utility>

#include <cuda.h>

#include "util/aot_serialize.h"
#include <cccl/c/aot.h>
#include <cccl/c/aot_diagnostics.h>
#include <cccl/c/types.h>

extern "C" CCCL_C_API void cccl_aot_buffer_free(void* buf)
{
  // Buffers handed out by *_serialize are allocated with new[] in the
  // matching cccl_aot::buffer_writer::release implementation.
  delete[] static_cast<char*>(buf);
}

namespace
{
// Per-thread last-error string. Set on an AoT failure, read via
// cccl_aot_last_error(). Cleared at the start of cccl_aot_validate_blob so a
// stale message from an earlier call is never reported for a later one.
thread_local std::string g_aot_last_error;

void set_aot_error(std::string msg)
{
  g_aot_last_error = std::move(msg);
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

extern "C" CCCL_C_API const char* cccl_aot_last_error(void)
{
  return g_aot_last_error.c_str();
}

extern "C" CCCL_C_API CUresult cccl_aot_validate_blob(const void* buf, size_t size)
try
{
  using namespace cccl::aot;
  g_aot_last_error.clear();

  if (buf == nullptr || size < sizeof(blob_header))
  {
    set_aot_error("AoT blob: buffer is null or smaller than the blob header");
    return CUDA_ERROR_INVALID_VALUE;
  }

  blob_header h{};
  std::memcpy(&h, buf, sizeof(h));

  if (std::memcmp(h.magic, k_blob_magic, sizeof(k_blob_magic)) != 0)
  {
    set_aot_error("AoT blob: bad magic (not a CCCL AoT blob)");
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (h.format_version != k_format_version)
  {
    set_aot_error(
      std::format("AoT blob: unsupported format version (blob={}, current={}); rebuild the blob with this cuda-cccl",
                  h.format_version,
                  k_format_version));
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (h.cccl_version != static_cast<uint64_t>(CCCL_C_PARALLEL_VERSION))
  {
    set_aot_error(
      std::format("AoT blob: CCCL C parallel ABI mismatch (blob={}, current={}); rebuild the blob with this cuda-cccl",
                  h.cccl_version,
                  CCCL_C_PARALLEL_VERSION));
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
        set_aot_error(std::format(
          "AoT blob targets sm_{} but the current device is sm_{}{}; a CUBIN payload is not compatible across "
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
  set_aot_error(std::string("AoT blob validation failed: ") + exc.what());
  return CUDA_ERROR_UNKNOWN;
}
