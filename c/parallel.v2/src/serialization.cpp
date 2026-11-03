//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <cstring>

#include <cccl/c/serialization.h>
#include <util/build_utils.h>
#include <util/serialization.h>

void cccl_serialization_v2_buffer_free(void* buf)
{
  delete[] static_cast<char*>(buf);
}

const char* cccl_serialization_v2_last_error(void)
{
  return cccl::serialization_v2::last_error_storage().c_str();
}

CUresult cccl_serialization_v2_validate_blob(const void* buf, size_t size)
try
{
  namespace sv2 = cccl::serialization_v2;
  sv2::clear_last_error();

  sv2::blob_header header;
  if (buf == nullptr || size < sizeof(header))
  {
    sv2::set_last_error("cccl.c.parallel.v2 serialization: truncated blob");
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::memcpy(&header, buf, sizeof(header));
  if (std::memcmp(header.magic, sv2::k_blob_magic, sizeof(sv2::k_blob_magic)) != 0)
  {
    sv2::set_last_error("cccl.c.parallel.v2 serialization: bad magic (not a v2 CCCL C parallel blob)");
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (auto error = cccl::detail::check_device_compute_capability(static_cast<int>(header.cc)); !error.empty())
  {
    sv2::set_last_error(error.c_str());
    return CUDA_ERROR_NO_BINARY_FOR_GPU;
  }
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  cccl::serialization_v2::set_last_error(exc.what());
  return CUDA_ERROR_UNKNOWN;
}
