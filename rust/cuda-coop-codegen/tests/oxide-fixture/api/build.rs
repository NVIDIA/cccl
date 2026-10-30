// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop_codegen::{Algorithm, Operation, oxide};

fn main() {
    let argument = std::env::args().nth(1);
    if matches!(
        argument.as_deref(),
        Some("--oxide-library-request-v1" | "--oxide-library-inputs-v1")
    ) {
        let checkout = std::path::Path::new(env!("CCCL_SOURCE"));
        let result = if argument.as_deref() == Some("--oxide-library-inputs-v1") {
            oxide::serve_inputs(checkout, std::io::stdin().lock(), std::io::stdout().lock())
        } else {
            oxide::serve(checkout, std::io::stdin().lock(), std::io::stdout().lock())
        };
        if let Err(error) = result {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=CCCL_SOURCE");
    let executable = std::env::current_exe().unwrap();
    for (algorithm, tag) in [
        (Algorithm::Direct, "DIRECT"),
        (Algorithm::Transpose, "TRANSPOSE"),
    ] {
        for (operation, name) in [
            (Operation::LoadFull, "LOAD_FULL"),
            (Operation::LoadFilled, "LOAD_FILLED"),
            (Operation::StoreFull, "STORE_FULL"),
            (Operation::StorePartial, "STORE_PARTIAL"),
        ] {
            let registration = oxide::registration(&executable, operation, algorithm).unwrap();
            println!("cargo::rustc-env=COOP_{tag}_{name}={registration}");
        }
    }
    for (operation, name) in [
        (Operation::LoadFullValidity, "LOAD_FULL"),
        (Operation::LoadMaskedValidity, "LOAD_MASKED"),
        (Operation::LoadFilledValidity, "LOAD_FILLED"),
        (Operation::StoreFullValidity, "STORE_FULL"),
        (Operation::StorePartialValidity, "STORE_PARTIAL"),
    ] {
        let registration = oxide::registration(&executable, operation, Algorithm::Direct).unwrap();
        println!("cargo::rustc-env=COOP_VALIDITY_{name}={registration}");
    }
}
