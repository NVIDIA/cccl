// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/__nvtx/nvtx.h>

#include <catch2/interfaces/catch_interfaces_capture.hpp>

namespace detail
{
struct nvtx_c2h_domain
{
  static constexpr const char* name = "C2H";
};

template <typename T>
class nvtx_fixture
{
#if _CCCL_HAS_NVTX3()
  const ::nvtx3::v1::scoped_range_in<nvtx_c2h_domain> nvtx_range{::Catch::getResultCapture().getCurrentTestName()};
#endif // _CCCL_HAS_NVTX3()
};
} // namespace detail
