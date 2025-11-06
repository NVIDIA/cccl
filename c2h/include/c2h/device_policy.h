// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <thrust/execution_policy.h>

#include <c2h/checked_allocator.cuh>

namespace c2h
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
static const auto device_policy        = THRUST_NS_QUALIFIER::cuda::par(checked_cuda_allocator<char>{});
static const auto nosync_device_policy = THRUST_NS_QUALIFIER::cuda::par_nosync(checked_cuda_allocator<char>{});
#else // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
static const auto device_policy        = THRUST_NS_QUALIFIER::device;
static const auto nosync_device_policy = THRUST_NS_QUALIFIER::device;
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
} // namespace c2h
