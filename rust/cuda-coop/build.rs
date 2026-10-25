// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Register the owning host provider for native Load, Store and Scan.
use cuda_coop_codegen::{
    Algorithm, Operation, ScanAlgorithm, ScanOperation, ScanOperator, ScanRecipe, oxide,
};

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
        (ScanOperation::ExclusiveSum, "EXCLUSIVE_ZERO"),
        (ScanOperation::Exclusive(ScanOperator::Sum), "EXCLUSIVE_SUM"),
        (
            ScanOperation::Exclusive(ScanOperator::Multiplies),
            "EXCLUSIVE_MULTIPLIES",
        ),
        (ScanOperation::Exclusive(ScanOperator::Min), "EXCLUSIVE_MIN"),
        (ScanOperation::Exclusive(ScanOperator::Max), "EXCLUSIVE_MAX"),
        (
            ScanOperation::Exclusive(ScanOperator::BitAnd),
            "EXCLUSIVE_BIT_AND",
        ),
        (
            ScanOperation::Exclusive(ScanOperator::BitOr),
            "EXCLUSIVE_BIT_OR",
        ),
        (
            ScanOperation::Exclusive(ScanOperator::BitXor),
            "EXCLUSIVE_BIT_XOR",
        ),
        (ScanOperation::Inclusive(ScanOperator::Sum), "INCLUSIVE_SUM"),
        (
            ScanOperation::Inclusive(ScanOperator::Multiplies),
            "INCLUSIVE_MULTIPLIES",
        ),
        (ScanOperation::Inclusive(ScanOperator::Min), "INCLUSIVE_MIN"),
        (ScanOperation::Inclusive(ScanOperator::Max), "INCLUSIVE_MAX"),
        (
            ScanOperation::Inclusive(ScanOperator::BitAnd),
            "INCLUSIVE_BIT_AND",
        ),
        (
            ScanOperation::Inclusive(ScanOperator::BitOr),
            "INCLUSIVE_BIT_OR",
        ),
        (
            ScanOperation::Inclusive(ScanOperator::BitXor),
            "INCLUSIVE_BIT_XOR",
        ),
    ] {
        for (recipe, prefix) in [
            (
                ScanRecipe::Block(operation, ScanAlgorithm::Raking),
                "RAKING",
            ),
            (
                ScanRecipe::Block(operation, ScanAlgorithm::RakingMemoize),
                "RAKING_MEMOIZE",
            ),
            (
                ScanRecipe::Block(operation, ScanAlgorithm::WarpScans),
                "WARP_SCANS",
            ),
            (ScanRecipe::Warp(operation), "WARP"),
        ] {
            let value =
                oxide::scan_registration(&executable, recipe).expect("cuda.coop scan registration");
            println!("cargo::rustc-env=COOP_SCAN_{prefix}_{name}={value}");
        }
    }
    for (operation, name) in [
        (Operation::LoadFullValidity, "LOAD_FULL"),
        (Operation::LoadMaskedValidity, "LOAD_MASKED"),
        (Operation::LoadFilledValidity, "LOAD_FILLED"),
        (Operation::StoreFullValidity, "STORE_FULL"),
        (Operation::StorePartialValidity, "STORE_PARTIAL"),
    ] {
        for (algorithm, prefix) in [
            (Algorithm::Striped, "STRIPED"),
            (Algorithm::Vectorize, "VECTORIZE"),
            (Algorithm::Transpose, "TRANSPOSE"),
            (Algorithm::WarpTranspose, "WARP_TRANSPOSE"),
            (
                Algorithm::WarpTransposeTimesliced,
                "WARP_TRANSPOSE_TIMESLICED",
            ),
        ] {
            let value = oxide::registration(&executable, operation, algorithm)
                .expect("cuda.coop algorithm registration");
            println!("cargo::rustc-env=COOP_{prefix}_VALIDITY_{name}={value}");
        }
    }
}
