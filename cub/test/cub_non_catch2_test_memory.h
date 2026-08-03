// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3

// TODO: Remove this header once test_device_batch_copy.cu is migrated to Catch2.
// Make all remaining non-Catch2 runtime tests run as CUB_SMALL.

#pragma once

// Non-Catch2 runtime tests must declare exactly one memory class. CMake checks
// this declaration while registering the test.

#define CUB_TEST_MEMORY_CLASS(MEMORY_CLASS) CUB_TEST_MEMORY_CLASS_##MEMORY_CLASS

#define CUB_TEST_MEMORY_CLASS_CUB_SMALL static_assert(true)
#define CUB_TEST_MEMORY_CLASS_CUB_LARGE static_assert(true)
