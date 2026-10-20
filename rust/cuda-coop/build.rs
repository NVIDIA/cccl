// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Register the owning host provider for native Direct operations.

fn main() {
    cuda_coop_codegen::oxide::build_script().expect("cuda.coop native provider registration");
}
