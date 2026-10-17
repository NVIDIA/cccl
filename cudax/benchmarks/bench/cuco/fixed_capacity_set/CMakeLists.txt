# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

file(GLOB bench_srcs CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

add_bench_dir(bench_targets ${bench_srcs})
