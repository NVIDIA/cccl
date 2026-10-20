// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop_codegen::{Algorithm, Operation, oxide};

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--oxide-library-request-v1") {
        if let Err(error) = oxide::serve(
            std::path::Path::new(env!("CCCL_SOURCE")),
            std::io::stdin().lock(),
            std::io::stdout().lock(),
        ) {
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
}
