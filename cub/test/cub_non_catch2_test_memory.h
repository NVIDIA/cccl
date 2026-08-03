// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

// Non-Catch2 runtime tests must declare exactly one memory class. CMake checks
// this declaration while registering the test.

#define CUB_TEST_MEMORY_CLASS(MEMORY_CLASS) CUB_TEST_MEMORY_CLASS_##MEMORY_CLASS

#define CUB_TEST_MEMORY_CLASS_CUB_SMALL static_assert(true)
// TODO: Remove CUB_LARGE once test_device_batch_copy.cu is migrated to Catch2.
// All remaining non-Catch2 runtime tests are small.
#define CUB_TEST_MEMORY_CLASS_CUB_LARGE static_assert(true)
