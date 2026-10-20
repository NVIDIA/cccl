// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Register the owning host provider for native block Load and Store.
use cuda_coop_codegen::{Algorithm, Operation, oxide};

fn main() {
    oxide::build_script().expect("cuda.coop native provider registration");
    if matches!(
        std::env::args().nth(1).as_deref(),
        Some("--oxide-library-request-v1" | "--oxide-library-inputs-v1")
    ) {
        return;
    }
    let executable = std::env::current_exe().expect("cuda.coop provider executable");
    for (operation, name) in [
        (Operation::LoadFullValidity, "LOAD_FULL"),
        (Operation::LoadMaskedValidity, "LOAD_MASKED"),
        (Operation::LoadFilledValidity, "LOAD_FILLED"),
        (Operation::StoreFullValidity, "STORE_FULL"),
        (Operation::StorePartialValidity, "STORE_PARTIAL"),
    ] {
        let value = oxide::registration(&executable, operation, Algorithm::Transpose)
            .expect("cuda.coop Transpose registration");
        println!("cargo::rustc-env=COOP_TRANSPOSE_VALIDITY_{name}={value}");
    }
}
