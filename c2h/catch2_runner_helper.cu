// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! This file includes CUDA-specific utilities for custom Catch2 main function when CMake is configured to build all
//! tests into a single executable. In this case, we have to have a CUDA target in the final Catch2 executable,
//! otherwise CMake confuses linker options and MSVC/RDC build fails.

#include "catch2_runner_helper.inl"
