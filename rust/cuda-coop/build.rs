// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Register the owning host provider for native cooperative operations.
use cuda_coop_codegen::{
    AdjacentDifferenceRecipe, Algorithm, DiscontinuityRecipe, Dtype, ExchangeMode, ExchangeRecipe,
    HistogramAlgorithm, Operation, RadixRankKeysRecipe, RadixSortKeysRecipe, RadixSortPairsRecipe,
    ReduceAlgorithm, ReduceBatchedLayout, ReduceBatchedRecipe, ReduceForm, ReduceOperator,
    ReduceRecipe, ScanAlgorithm, ScanOperation, ScanOperator, ScanRecipe, ShuffleRecipe, oxide,
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
    for bits in 1..=8 {
        for (recipe, order) in [
            (RadixRankKeysRecipe::Ascending(bits), "ASCENDING"),
            (RadixRankKeysRecipe::Descending(bits), "DESCENDING"),
        ] {
            let value = oxide::radix_rank_keys_registration(&executable, recipe)
                .expect("cuda.coop radix rank keys registration");
            println!("cargo::rustc-env=COOP_RADIX_RANK_KEYS_{order}_{bits}={value}");
        }
    }
    for (recipe, name) in [
        (RadixSortKeysRecipe::Ascending, "ASCENDING"),
        (RadixSortKeysRecipe::Descending, "DESCENDING"),
    ] {
        let value = oxide::radix_sort_keys_registration(&executable, recipe)
            .expect("cuda.coop radix sort keys registration");
        println!("cargo::rustc-env=COOP_RADIX_SORT_KEYS_{name}={value}");
    }
    for (recipe, order) in [
        (RadixSortPairsRecipe::Ascending, "ASCENDING"),
        (RadixSortPairsRecipe::Descending, "DESCENDING"),
    ] {
        for (dtype, name) in [
            (Dtype::I8, "I8"),
            (Dtype::U8, "U8"),
            (Dtype::I16, "I16"),
            (Dtype::U16, "U16"),
            (Dtype::I32, "I32"),
            (Dtype::U32, "U32"),
            (Dtype::I64, "I64"),
            (Dtype::U64, "U64"),
            (Dtype::F32, "F32"),
            (Dtype::F64, "F64"),
        ] {
            let value = oxide::radix_sort_pairs_registration(&executable, recipe, dtype)
                .expect("cuda.coop radix sort pairs registration");
            println!("cargo::rustc-env=COOP_RADIX_SORT_PAIRS_{order}_{name}={value}");
        }
    }
    for (algorithm, name) in [
        (HistogramAlgorithm::Atomic, "ATOMIC"),
        (HistogramAlgorithm::Sort, "SORT"),
    ] {
        let value = oxide::histogram_registration_template(&executable, algorithm)
            .expect("cuda.coop histogram registration template");
        println!("cargo::rustc-env=COOP_HISTOGRAM_{name}={value}");
    }
    for (recipe, name) in [
        (DiscontinuityRecipe::Heads, "HEADS"),
        (DiscontinuityRecipe::HeadsPredecessor, "HEADS_PREDECESSOR"),
        (DiscontinuityRecipe::Tails, "TAILS"),
        (DiscontinuityRecipe::TailsSuccessor, "TAILS_SUCCESSOR"),
        (DiscontinuityRecipe::HeadsAndTails, "HEADS_AND_TAILS"),
        (
            DiscontinuityRecipe::HeadsAndTailsPredecessor,
            "HEADS_AND_TAILS_PREDECESSOR",
        ),
        (
            DiscontinuityRecipe::HeadsAndTailsSuccessor,
            "HEADS_AND_TAILS_SUCCESSOR",
        ),
        (
            DiscontinuityRecipe::HeadsAndTailsBoth,
            "HEADS_AND_TAILS_BOTH",
        ),
    ] {
        let value = oxide::discontinuity_registration(&executable, recipe)
            .expect("cuda.coop discontinuity registration");
        println!("cargo::rustc-env=COOP_DISCONTINUITY_{name}={value}");
    }
    for (recipe, name) in [
        (AdjacentDifferenceRecipe::LeftFull, "LEFT_FULL"),
        (
            AdjacentDifferenceRecipe::LeftFullPredecessor,
            "LEFT_FULL_PREDECESSOR",
        ),
        (AdjacentDifferenceRecipe::LeftPartial, "LEFT_PARTIAL"),
        (
            AdjacentDifferenceRecipe::LeftPartialPredecessor,
            "LEFT_PARTIAL_PREDECESSOR",
        ),
        (AdjacentDifferenceRecipe::RightFull, "RIGHT_FULL"),
        (
            AdjacentDifferenceRecipe::RightFullSuccessor,
            "RIGHT_FULL_SUCCESSOR",
        ),
        (AdjacentDifferenceRecipe::RightPartial, "RIGHT_PARTIAL"),
    ] {
        let value = oxide::adjacent_difference_registration(&executable, recipe)
            .expect("cuda.coop adjacent difference registration");
        println!("cargo::rustc-env=COOP_ADJACENT_DIFFERENCE_{name}={value}");
    }
    for (recipe, name) in [(ShuffleRecipe::Down, "DOWN"), (ShuffleRecipe::Up, "UP")] {
        let value = oxide::shuffle_registration(&executable, recipe)
            .expect("cuda.coop shuffle registration");
        println!("cargo::rustc-env=COOP_SHUFFLE_{name}={value}");
    }
    for (mode, name) in [
        (ExchangeMode::StripedToBlocked, "STRIPED_TO_BLOCKED"),
        (ExchangeMode::BlockedToStriped, "BLOCKED_TO_STRIPED"),
    ] {
        for (recipe, group) in [
            (ExchangeRecipe::Block(mode), "BLOCK"),
            (ExchangeRecipe::Warp(mode), "WARP"),
        ] {
            let value = oxide::exchange_registration(&executable, recipe)
                .expect("cuda.coop exchange registration");
            println!("cargo::rustc-env=COOP_EXCHANGE_{group}_{name}={value}");
        }
    }

    for (operator, name) in [
        (ReduceOperator::Sum, "SUM"),
        (ReduceOperator::Multiplies, "MULTIPLIES"),
        (ReduceOperator::Min, "MIN"),
        (ReduceOperator::Max, "MAX"),
        (ReduceOperator::BitAnd, "BIT_AND"),
        (ReduceOperator::BitOr, "BIT_OR"),
        (ReduceOperator::BitXor, "BIT_XOR"),
    ] {
        for (layout, suffix) in [
            (ReduceBatchedLayout::Striped, "STRIPED"),
            (ReduceBatchedLayout::Blocked, "BLOCKED"),
        ] {
            let value = oxide::reduce_batched_registration(
                &executable,
                ReduceBatchedRecipe::new(operator, layout),
            )
            .expect("cuda.coop batched reduce registration");
            println!("cargo::rustc-env=COOP_REDUCE_BATCHED_{name}_{suffix}={value}");
        }
        for (form, suffix) in [
            (ReduceForm::Scalar, "SCALAR"),
            (ReduceForm::ScalarPrefix, "PREFIX"),
            (ReduceForm::Payload, "PAYLOAD"),
        ] {
            for (recipe, prefix) in [
                (
                    ReduceRecipe::Block(operator, ReduceAlgorithm::RakingCommutativeOnly, form),
                    "RAKING_COMMUTATIVE_ONLY",
                ),
                (
                    ReduceRecipe::Block(operator, ReduceAlgorithm::Raking, form),
                    "RAKING",
                ),
                (
                    ReduceRecipe::Block(operator, ReduceAlgorithm::WarpReductions, form),
                    "WARP_REDUCTIONS",
                ),
                (ReduceRecipe::Warp(operator, form), "WARP"),
            ] {
                let value = oxide::reduce_registration(&executable, recipe)
                    .expect("cuda.coop reduce registration");
                println!("cargo::rustc-env=COOP_REDUCE_{prefix}_{name}_{suffix}={value}");
            }
        }
    }
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
