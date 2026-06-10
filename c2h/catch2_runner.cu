// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! This file includes a custom Catch2 main function when CMake is configured to build all tests into a single
//! executable.

#define C2H_CONFIG_MAIN
#define C2H_EXCLUDE_CATCH2_HELPER_IMPL
#include <c2h/catch2_main.h>
