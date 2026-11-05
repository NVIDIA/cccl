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
  return cccl::detail::last_error_storage().c_str();
}

CUresult cccl_serialization_v2_validate_blob(const void* buf, size_t size)
{
  return cccl::detail::guard("cccl_serialization_v2_validate_blob", [&] {
    namespace sv2 = cccl::serialization_v2;

    sv2::blob_header header;
    if (buf == nullptr || size < sizeof(header))
    {
      throw cccl::detail::cccl_error(CUDA_ERROR_INVALID_VALUE, "cccl.c.parallel.v2 serialization: truncated blob");
    }
    std::memcpy(&header, buf, sizeof(header));
    if (std::memcmp(header.magic, sv2::k_blob_magic, sizeof(sv2::k_blob_magic)) != 0)
    {
      throw cccl::detail::cccl_error(
        CUDA_ERROR_INVALID_VALUE, "cccl.c.parallel.v2 serialization: bad magic (not a v2 CCCL C parallel blob)");
    }
    if (auto error = cccl::detail::check_device_compute_capability(static_cast<int>(header.cc)); !error.empty())
    {
      throw cccl::detail::cccl_error(CUDA_ERROR_NO_BINARY_FOR_GPU, error);
    }
  });
}
