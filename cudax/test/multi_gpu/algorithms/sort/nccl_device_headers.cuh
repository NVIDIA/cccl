// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CUDAX_TEST_MULTI_GPU_SORT_NCCL_DEVICE_HEADERS_CUH
#define CUDAX_TEST_MULTI_GPU_SORT_NCCL_DEVICE_HEADERS_CUH

// The capability-interposition test must include the SDK before sort. Treat
// these external headers as system headers even when testing CCCL with warnings
// enabled; the packaged NCCL headers contain unused parameters/packed fields.
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC system_header
#endif

#include <nccl.h>
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 2) && __has_include(<nccl_device.h>)
#  include <nccl_device.h>
#endif

#endif // CUDAX_TEST_MULTI_GPU_SORT_NCCL_DEVICE_HEADERS_CUH
