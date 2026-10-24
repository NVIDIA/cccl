// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Real native compiler coverage; ordinary host tests need no CUDA installation.

use cuda_coop_codegen::{
    AdjacentDifferenceRecipe, AdjacentDifferenceRequest, Algorithm, Dtype, ExchangeMode,
    ExchangeRecipe, ExchangeRequest, Layout, NativeRequest, Operation, Provider, ReduceAlgorithm,
    ReduceBatchedLayout, ReduceBatchedRecipe, ReduceBatchedRequest, ReduceForm, ReduceOperator,
    ReduceRequest, Request, ScanAlgorithm, ScanOperation, ScanOperator, ScanRequest, ShuffleRecipe,
    ShuffleRequest,
};
use std::{env, path::Path, time::Instant};

#[test]
#[ignore = "requires CUDA_HOME and a Linux CUDA toolkit with NVRTC LTO support"]
fn native_mixed_adjacent_difference_recipes() -> Result<(), Box<dyn std::error::Error>> {
    use AdjacentDifferenceRecipe::*;
    let toolkit = env::var("CUDA_HOME")?;
    let target = env::var("CUDA_COOP_TEST_TARGET").unwrap_or_else(|_| "compute_120".into());
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provider = Provider::from_checkout(&checkout, &toolkit)?;
    let mut requests = Vec::new();
    // Distribute all types and seven overloads over small, odd and XYZ Blocks;
    // N9 crosses the packed-validity byte boundary without a Cartesian matrix.
    for (dtype, block, items, recipe) in [
        (Dtype::I8, [1, 1, 1], 1, LeftFull),
        (Dtype::U8, [1, 1, 1], 1, RightFull),
        (Dtype::I16, [33, 1, 1], 1, LeftFullPredecessor),
        (Dtype::U16, [4, 3, 2], 4, RightFullSuccessor),
        (Dtype::I32, [33, 1, 1], 9, LeftPartial),
        (Dtype::U32, [64, 1, 1], 4, RightPartial),
        (Dtype::I64, [4, 3, 2], 4, LeftPartialPredecessor),
        (Dtype::U64, [64, 1, 1], 1, LeftFull),
        (Dtype::F32, [128, 1, 1], 4, RightFull),
        (Dtype::F64, [4, 3, 2], 9, LeftPartialPredecessor),
    ] {
        requests.push(NativeRequest::AdjacentDifference(
            AdjacentDifferenceRequest::block(dtype, recipe, block, items)?,
        ));
    }
    requests.extend([
        NativeRequest::LoadStore(Request::new_typed(
            Dtype::I32,
            Operation::LoadFullValidity,
            Algorithm::Direct,
            64,
            4,
        )?),
        NativeRequest::Scan(ScanRequest::warp(
            Dtype::I32,
            ScanOperation::ExclusiveSum,
            [64, 1, 1],
            8,
        )?),
        NativeRequest::Reduce(ReduceRequest::warp(
            Dtype::I32,
            ReduceOperator::Sum,
            ReduceForm::Payload,
            [64, 1, 1],
            8,
            4,
        )?),
        NativeRequest::ReduceBatched(ReduceBatchedRequest::warp(
            Dtype::I32,
            ReduceBatchedRecipe::new(ReduceOperator::Sum, ReduceBatchedLayout::Striped),
            [64, 1, 1],
            8,
            4,
        )?),
        NativeRequest::Exchange(ExchangeRequest::warp(
            Dtype::I32,
            ExchangeRecipe::Warp(ExchangeMode::StripedToBlocked),
            [64, 1, 1],
            32,
            4,
        )?),
        NativeRequest::Shuffle(ShuffleRequest::block(
            Dtype::I32,
            ShuffleRecipe::Down,
            [64, 1, 1],
            4,
        )?),
    ]);
    let artifact = provider.compile_native(&requests, &target)?;
    assert_eq!(artifact.specializations.len(), 16);
    assert!(!artifact.ltoir.is_empty());
    let mut identities = std::collections::HashSet::new();
    let mut symbols = std::collections::HashSet::new();
    for (request, entry) in requests.iter().zip(&artifact.specializations) {
        assert!(identities.insert(entry.identity));
        assert!(symbols.insert(&entry.symbol));
        if matches!(request, NativeRequest::LoadStore(_)) {
            assert!(entry.scratch_free);
            assert_eq!(entry.scratch.size, 0);
        } else {
            assert_eq!(entry.cub_layout, entry.scratch);
            assert!(entry.scratch.size > 0 && entry.scratch.alignment.is_power_of_two());
            assert!(!entry.scratch_free && !entry.mutates_payload);
        }
        eprintln!(
            "{request:?}: CUB {:?}, scratch {:?}",
            entry.cub_layout, entry.scratch
        );
    }
    // Both endpoint planes belong to CUB TempStorage, even for one direction.
    assert_eq!(
        artifact.specializations[0].scratch,
        Layout {
            size: 2,
            alignment: 1
        }
    );
    assert_eq!(
        artifact.specializations[9].scratch,
        Layout {
            size: 384,
            alignment: 8
        }
    );
    assert!(
        provider
            .compile_native(&[requests[0], requests[0]], &target)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    eprintln!(
        "{} mixed Adjacent Difference/Load/Scan/Reduce/batched Reduce/Exchange/Shuffle specializations, {} LTO bytes",
        requests.len(),
        artifact.ltoir.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA_HOME and a Linux CUDA toolkit with NVRTC LTO support"]
fn native_mixed_shuffle_recipes() -> Result<(), Box<dyn std::error::Error>> {
    use ShuffleRecipe::{Down, Up};
    let toolkit = env::var("CUDA_HOME")?;
    let target = env::var("CUDA_COOP_TEST_TARGET").unwrap_or_else(|_| "compute_120".into());
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provider = Provider::from_checkout(&checkout, &toolkit)?;
    let mut requests = Vec::new();
    // Cover all numeric types, both empty-result directions, cross-thread
    // shifts, XYZ and a second validity byte without a Cartesian matrix.
    for (dtype, block, items, recipe) in [
        (Dtype::I8, [1, 1, 1], 1, Down),
        (Dtype::U8, [1, 1, 1], 1, Up),
        (Dtype::I16, [33, 1, 1], 1, Down),
        (Dtype::U16, [4, 3, 2], 4, Up),
        (Dtype::I32, [1, 1, 1], 4, Down),
        (Dtype::U32, [33, 1, 1], 9, Up),
        (Dtype::I64, [64, 1, 1], 4, Down),
        (Dtype::U64, [4, 3, 2], 1, Up),
        (Dtype::F32, [128, 1, 1], 4, Down),
        (Dtype::F64, [4, 3, 2], 9, Up),
        (Dtype::I32, [33, 1, 1], 9, Down),
        (Dtype::I32, [33, 1, 1], 9, Up),
    ] {
        requests.push(NativeRequest::Shuffle(ShuffleRequest::block(
            dtype, recipe, block, items,
        )?));
    }
    requests.extend([
        NativeRequest::LoadStore(Request::new_typed(
            Dtype::I32,
            Operation::LoadFullValidity,
            Algorithm::Direct,
            64,
            4,
        )?),
        NativeRequest::Scan(ScanRequest::warp(
            Dtype::I32,
            ScanOperation::ExclusiveSum,
            [64, 1, 1],
            8,
        )?),
        NativeRequest::Reduce(ReduceRequest::warp(
            Dtype::I32,
            ReduceOperator::Sum,
            ReduceForm::Payload,
            [64, 1, 1],
            8,
            4,
        )?),
        NativeRequest::ReduceBatched(ReduceBatchedRequest::warp(
            Dtype::I32,
            ReduceBatchedRecipe::new(ReduceOperator::Sum, ReduceBatchedLayout::Striped),
            [64, 1, 1],
            8,
            4,
        )?),
        NativeRequest::Exchange(ExchangeRequest::warp(
            Dtype::I32,
            ExchangeRecipe::Warp(ExchangeMode::StripedToBlocked),
            [64, 1, 1],
            32,
            4,
        )?),
    ]);
    let artifact = provider.compile_native(&requests, &target)?;
    assert_eq!(artifact.specializations.len(), 17);
    assert!(!artifact.ltoir.is_empty());
    let mut identities = std::collections::HashSet::new();
    let mut symbols = std::collections::HashSet::new();
    for (request, entry) in requests.iter().zip(&artifact.specializations) {
        assert!(identities.insert(entry.identity));
        assert!(symbols.insert(&entry.symbol));
        if matches!(request, NativeRequest::LoadStore(_)) {
            assert!(entry.scratch_free);
            assert_eq!(entry.scratch.size, 0);
        } else {
            assert_eq!(entry.cub_layout, entry.scratch);
            assert!(entry.scratch.size > 0 && entry.scratch.alignment.is_power_of_two());
            assert!(!entry.scratch_free && !entry.mutates_payload);
        }
        eprintln!(
            "{request:?}: CUB {:?}, scratch {:?}",
            entry.cub_layout, entry.scratch
        );
    }
    // CUB scratch still exists for Block1/N1 even though the result is invalid.
    assert_eq!(
        artifact.specializations[0].scratch,
        Layout {
            size: 1,
            alignment: 1
        }
    );
    assert_eq!(
        artifact.specializations[1].scratch,
        Layout {
            size: 1,
            alignment: 1
        }
    );
    assert_eq!(
        artifact.specializations[9].scratch,
        Layout {
            size: 192,
            alignment: 8
        }
    );
    assert!(
        provider
            .compile_native(&[requests[0], requests[0]], &target)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    eprintln!(
        "{} mixed Shuffle/Load/Scan/Reduce/batched Reduce/Exchange specializations, {} LTO bytes",
        requests.len(),
        artifact.ltoir.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA_HOME and a Linux CUDA toolkit with NVRTC LTO support"]
fn native_warp_shape_identity() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = env::var("CUDA_HOME")?;
    let target = env::var("CUDA_COOP_TEST_TARGET").unwrap_or_else(|_| "compute_120".into());
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provider = Provider::from_checkout(&checkout, &toolkit)?;
    let mut requests = Vec::new();
    // Two shapes have the same total threads. Warp bodies can otherwise be
    // identical even though each root needs a distinct specialization record.
    for block in [[64, 1, 1], [128, 1, 1], [8, 4, 2]] {
        requests.extend([
            NativeRequest::LoadStore(
                Request::new_typed_with_shape(
                    Dtype::I32,
                    Operation::LoadMaskedValidity,
                    Algorithm::Transpose,
                    block,
                    4,
                )?
                .with_warp_width(32)?,
            ),
            NativeRequest::Scan(ScanRequest::warp(
                Dtype::I32,
                ScanOperation::ExclusiveSum,
                block,
                32,
            )?),
            NativeRequest::Reduce(ReduceRequest::warp(
                Dtype::I32,
                ReduceOperator::Sum,
                ReduceForm::Payload,
                block,
                32,
                4,
            )?),
            NativeRequest::ReduceBatched(ReduceBatchedRequest::warp(
                Dtype::I32,
                ReduceBatchedRecipe::new(ReduceOperator::Sum, ReduceBatchedLayout::Striped),
                block,
                32,
                4,
            )?),
        ]);
    }
    let artifact = provider.compile_native(&requests, &target)?;
    assert_eq!(artifact.specializations.len(), 12);
    assert!(!artifact.ltoir.is_empty());
    let mut identities = std::collections::HashSet::new();
    let mut symbols = std::collections::HashSet::new();
    let parameters = |symbol: &str| {
        artifact
            .source
            .split_once(&format!("void {symbol}("))
            .expect("compiled native wrapper")
            .1
            .split_once(") {")
            .expect("native parameter list")
            .0
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    for (index, entry) in artifact.specializations.iter().enumerate() {
        assert!(identities.insert(entry.identity));
        assert!(symbols.insert(&entry.symbol));
        let first = &artifact.specializations[index % 4];
        assert_eq!(entry.cub_layout, first.cub_layout);
        assert_eq!(entry.scratch, first.scratch);
        assert!(entry.scratch.size > 0);
        assert!(!entry.scratch_free && !entry.mutates_payload);
        assert_eq!(parameters(&entry.symbol), parameters(&first.symbol));
        eprintln!(
            "{:?}: CUB {:?}, scratch {:?}",
            requests[index], entry.cub_layout, entry.scratch
        );
    }
    requests.reverse();
    let reordered = provider.compile_native(&requests, &target)?;
    assert_eq!(
        reordered.specializations.len(),
        artifact.specializations.len()
    );
    assert!(!reordered.ltoir.is_empty());
    assert_ne!(reordered.identity, artifact.identity);
    for (entry, previous) in reordered
        .specializations
        .iter()
        .zip(artifact.specializations.iter().rev())
    {
        assert_eq!(entry.identity, previous.identity);
        assert_eq!(entry.symbol, previous.symbol);
        assert_eq!(entry.cub_layout, previous.cub_layout);
        assert_eq!(entry.scratch, previous.scratch);
        assert_eq!(entry.scratch_free, previous.scratch_free);
        assert_eq!(entry.mutates_payload, previous.mutates_payload);
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA_HOME and a Linux CUDA toolkit with NVRTC LTO support"]
fn native_mixed_exchange_recipes() -> Result<(), Box<dyn std::error::Error>> {
    use ExchangeMode::{BlockedToStriped, StripedToBlocked};
    let toolkit = env::var("CUDA_HOME")?;
    let target = env::var("CUDA_COOP_TEST_TARGET").unwrap_or_else(|_| "compute_120".into());
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provider = Provider::from_checkout(&checkout, &toolkit)?;
    // Ten numeric types, both directions, all Warp widths, XYZ and padded
    // extents are distributed over rows rather than taking their product.
    let mut requests = Vec::new();
    for (dtype, block, width, items, mode) in [
        (Dtype::I8, [33, 1, 1], None, 1, StripedToBlocked),
        (Dtype::U8, [64, 1, 1], Some(1), 9, BlockedToStriped),
        (Dtype::I16, [4, 3, 2], None, 4, BlockedToStriped),
        (Dtype::U16, [4, 4, 4], Some(2), 4, StripedToBlocked),
        (Dtype::I32, [128, 1, 1], None, 4, StripedToBlocked),
        (Dtype::U32, [64, 1, 1], Some(4), 8, BlockedToStriped),
        (Dtype::I64, [1, 1, 1], None, 1, BlockedToStriped),
        (Dtype::U64, [64, 1, 1], Some(8), 4, StripedToBlocked),
        (Dtype::F32, [64, 1, 1], Some(16), 1, BlockedToStriped),
        (Dtype::F64, [8, 4, 2], Some(32), 4, StripedToBlocked),
        (Dtype::I32, [33, 1, 1], None, 8, BlockedToStriped),
        (Dtype::I32, [64, 1, 1], Some(32), 4, StripedToBlocked),
        (Dtype::I32, [128, 1, 1], Some(32), 4, StripedToBlocked),
    ] {
        requests.push(NativeRequest::Exchange(match width {
            None => ExchangeRequest::block(dtype, ExchangeRecipe::Block(mode), block, items)?,
            Some(width) => {
                ExchangeRequest::warp(dtype, ExchangeRecipe::Warp(mode), block, width, items)?
            }
        }));
    }
    requests.push(NativeRequest::LoadStore(Request::new_typed(
        Dtype::I32,
        Operation::LoadFullValidity,
        Algorithm::Direct,
        64,
        4,
    )?));
    requests.push(NativeRequest::Scan(ScanRequest::warp(
        Dtype::I32,
        ScanOperation::ExclusiveSum,
        [64, 1, 1],
        8,
    )?));
    requests.push(NativeRequest::Reduce(ReduceRequest::warp(
        Dtype::I32,
        ReduceOperator::Sum,
        ReduceForm::Payload,
        [64, 1, 1],
        8,
        4,
    )?));
    requests.push(NativeRequest::ReduceBatched(ReduceBatchedRequest::warp(
        Dtype::I32,
        ReduceBatchedRecipe::new(ReduceOperator::Sum, ReduceBatchedLayout::Striped),
        [64, 1, 1],
        8,
        4,
    )?));
    let artifact = provider.compile_native(&requests, &target)?;
    assert_eq!(artifact.specializations.len(), 17);
    assert!(!artifact.ltoir.is_empty());
    let mut identities = std::collections::HashSet::new();
    let mut symbols = std::collections::HashSet::new();
    for (request, entry) in requests.iter().zip(&artifact.specializations) {
        assert!(identities.insert(entry.identity));
        assert!(symbols.insert(&entry.symbol));
        if matches!(request, NativeRequest::LoadStore(_)) {
            assert!(entry.scratch_free);
            assert_eq!(entry.scratch.size, 0);
        } else {
            assert_eq!(entry.cub_layout, entry.scratch);
            assert!(entry.scratch.size > 0 && entry.scratch.alignment.is_power_of_two());
            assert!(!entry.scratch_free && !entry.mutates_payload);
        }
        eprintln!(
            "{request:?}: CUB {:?}, scratch {:?}",
            entry.cub_layout, entry.scratch
        );
    }
    // Actual queried layouts include Warp sentinel/padding and Block padding.
    assert_eq!(
        artifact.specializations[5].scratch,
        Layout {
            size: 136,
            alignment: 4
        }
    );
    assert_eq!(
        artifact.specializations[10].scratch,
        Layout {
            size: 1088,
            alignment: 16
        }
    );
    eprintln!(
        "{} mixed Exchange/Load/Scan/Reduce/batched Reduce specializations, {} LTO bytes",
        requests.len(),
        artifact.ltoir.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA_HOME and a Linux CUDA toolkit with NVRTC LTO support"]
fn native_mixed_reduce_batched_recipes() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = env::var("CUDA_HOME")?;
    let target = env::var("CUDA_COOP_TEST_TARGET").unwrap_or_else(|_| "compute_120".into());
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provider = Provider::from_checkout(&checkout, &toolkit)?;
    let mut requests = Vec::new();
    for (width, batches, layout) in [
        (32, 1, ReduceBatchedLayout::Striped),
        (8, 4, ReduceBatchedLayout::Striped),
        (8, 10, ReduceBatchedLayout::Striped),
        (8, 10, ReduceBatchedLayout::Blocked),
        (1, 9, ReduceBatchedLayout::Blocked),
        (8, 65, ReduceBatchedLayout::Striped),
        (8, 65, ReduceBatchedLayout::Blocked),
    ] {
        requests.push(NativeRequest::ReduceBatched(ReduceBatchedRequest::warp(
            Dtype::I32,
            ReduceBatchedRecipe::new(ReduceOperator::Sum, layout),
            [128, 1, 1],
            width,
            batches,
        )?));
    }
    // Spread types/operators/widths over cases rather than taking their product.
    for (index, (dtype, op)) in [
        (Dtype::I8, ReduceOperator::Sum),
        (Dtype::U8, ReduceOperator::Multiplies),
        (Dtype::I16, ReduceOperator::Min),
        (Dtype::U16, ReduceOperator::Max),
        (Dtype::I32, ReduceOperator::BitAnd),
        (Dtype::U32, ReduceOperator::BitOr),
        (Dtype::I64, ReduceOperator::BitXor),
        (Dtype::U64, ReduceOperator::Sum),
        (Dtype::F32, ReduceOperator::Min),
        (Dtype::F64, ReduceOperator::Max),
    ]
    .into_iter()
    .enumerate()
    {
        requests.push(NativeRequest::ReduceBatched(ReduceBatchedRequest::warp(
            dtype,
            ReduceBatchedRecipe::new(
                op,
                if index % 2 == 0 {
                    ReduceBatchedLayout::Striped
                } else {
                    ReduceBatchedLayout::Blocked
                },
            ),
            [4, 4, 4],
            [1, 2, 4, 8, 16, 32][index % 6],
            [1, 4, 10, 9, 65][index % 5],
        )?));
    }
    requests.push(NativeRequest::LoadStore(Request::new_typed(
        Dtype::I32,
        Operation::LoadFullValidity,
        Algorithm::Direct,
        64,
        4,
    )?));
    requests.push(NativeRequest::Scan(ScanRequest::warp(
        Dtype::I32,
        ScanOperation::ExclusiveSum,
        [4, 4, 4],
        8,
    )?));
    requests.push(NativeRequest::Reduce(ReduceRequest::warp(
        Dtype::I32,
        ReduceOperator::Sum,
        ReduceForm::Payload,
        [4, 4, 4],
        8,
        4,
    )?));
    let artifact = provider.compile_native(&requests, &target)?;
    assert_eq!(artifact.specializations.len(), requests.len());
    assert!(!artifact.ltoir.is_empty());
    let mut identities = std::collections::HashSet::new();
    let mut symbols = std::collections::HashSet::new();
    for (request, entry) in requests.iter().zip(&artifact.specializations) {
        assert!(identities.insert(entry.identity));
        assert!(symbols.insert(&entry.symbol));
        if matches!(request, NativeRequest::LoadStore(_)) {
            assert!(entry.scratch_free);
            assert_eq!(entry.scratch.size, 0);
        } else {
            assert_eq!(entry.cub_layout, entry.scratch);
            assert!(entry.scratch.size > 0 && entry.scratch.alignment.is_power_of_two());
            assert!(!entry.scratch_free && !entry.mutates_payload);
        }
        eprintln!(
            "{request:?}: CUB {:?}, scratch {:?}",
            entry.cub_layout, entry.scratch
        );
    }
    let single = provider.compile_native(&requests[..1], &target)?;
    assert_eq!(
        single.specializations[0].identity,
        artifact.specializations[0].identity
    );
    assert_eq!(
        single.specializations[0].symbol,
        artifact.specializations[0].symbol
    );
    assert_eq!(
        single.specializations[0].scratch,
        artifact.specializations[0].scratch
    );
    assert!(
        provider
            .compile_native(&[requests[0]; 2], &target)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    eprintln!(
        "{} mixed batched Reduce/Load/Scan/Reduce specializations, {} LTO bytes",
        requests.len(),
        artifact.ltoir.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA_HOME and a Linux CUDA toolkit with NVRTC LTO support"]
fn native_mixed_reduce_recipes() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = env::var("CUDA_HOME")?;
    let target = env::var("CUDA_COOP_TEST_TARGET").unwrap_or_else(|_| "compute_120".into());
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provider = Provider::from_checkout(&checkout, &toolkit)?;
    let requests = [1, 4].map(|items| {
        ReduceRequest::block(
            Dtype::I32,
            ReduceOperator::Sum,
            ReduceAlgorithm::WarpReductions,
            ReduceForm::Payload,
            [128, 1, 1],
            items,
        )
        .map(NativeRequest::Reduce)
    });
    let mut requests = requests
        .into_iter()
        .collect::<cuda_coop_codegen::Result<Vec<_>>>()?;
    // Distribute types, operators, forms and algorithms rather than multiplying
    // independent dimensions. Payload extents cross the packed-mask byte boundary.
    let types = [
        Dtype::I8,
        Dtype::U8,
        Dtype::I16,
        Dtype::U16,
        Dtype::I32,
        Dtype::U32,
        Dtype::I64,
        Dtype::U64,
        Dtype::F32,
        Dtype::F64,
    ];
    let operators = [
        ReduceOperator::Sum,
        ReduceOperator::Multiplies,
        ReduceOperator::BitAnd,
        ReduceOperator::BitOr,
        ReduceOperator::BitXor,
        ReduceOperator::Min,
        ReduceOperator::Max,
    ];
    let algorithms = [
        ReduceAlgorithm::RakingCommutativeOnly,
        ReduceAlgorithm::Raking,
        ReduceAlgorithm::WarpReductions,
    ];
    let forms = [
        ReduceForm::Scalar,
        ReduceForm::ScalarPrefix,
        ReduceForm::Payload,
    ];
    for (index, dtype) in types.into_iter().enumerate() {
        let operator = match dtype {
            Dtype::F32 => ReduceOperator::Min,
            Dtype::F64 => ReduceOperator::Max,
            _ => operators[index % operators.len()],
        };
        let form = forms[index % forms.len()];
        requests.push(NativeRequest::Reduce(ReduceRequest::block(
            dtype,
            operator,
            algorithms[index % algorithms.len()],
            form,
            if index % 2 == 0 {
                [4, 4, 4]
            } else {
                [11, 3, 1]
            },
            if form == ReduceForm::Payload {
                if index % 2 == 0 { 4 } else { 9 }
            } else {
                1
            },
        )?));
    }
    for (index, width) in [1, 2, 4, 8, 16].into_iter().chain(17..=32).enumerate() {
        let dtype = types[index % types.len()];
        let operator = match dtype {
            Dtype::F32 => ReduceOperator::Min,
            Dtype::F64 => ReduceOperator::Max,
            _ => operators[index % operators.len()],
        };
        let form = forms[index % forms.len()];
        requests.push(NativeRequest::Reduce(ReduceRequest::warp(
            dtype,
            operator,
            form,
            [4, 4, 4],
            width,
            if form == ReduceForm::Payload {
                if index % 2 == 0 { 4 } else { 9 }
            } else {
                1
            },
        )?));
    }
    requests.push(NativeRequest::LoadStore(Request::new_typed(
        Dtype::I32,
        Operation::LoadFullValidity,
        Algorithm::Direct,
        64,
        4,
    )?));
    requests.push(NativeRequest::Scan(ScanRequest::block(
        Dtype::I32,
        ScanOperation::ExclusiveSum,
        ScanAlgorithm::Raking,
        [4, 4, 4],
        4,
    )?));
    let artifact = provider.compile_native(&requests, &target)?;
    assert_eq!(artifact.specializations.len(), requests.len());
    assert!(!artifact.ltoir.is_empty());
    let mut identities = std::collections::HashSet::new();
    let mut symbols = std::collections::HashSet::new();
    for (request, entry) in requests.iter().zip(&artifact.specializations) {
        assert!(identities.insert(entry.identity));
        assert!(symbols.insert(&entry.symbol));
        if matches!(request, NativeRequest::LoadStore(_)) {
            assert!(entry.scratch_free);
            assert_eq!(entry.scratch.size, 0);
        } else {
            assert_eq!(entry.cub_layout, entry.scratch);
            assert!(entry.scratch.size > 0 && entry.scratch.alignment.is_power_of_two());
            assert!(!entry.scratch_free && !entry.mutates_payload);
        }
        eprintln!(
            "{request:?}: CUB {:?}, scratch {:?}",
            entry.cub_layout, entry.scratch
        );
    }
    let single = provider.compile_native(&requests[..1], &target)?;
    assert_eq!(
        single.specializations[0].identity,
        artifact.specializations[0].identity
    );
    assert_eq!(
        single.specializations[0].symbol,
        artifact.specializations[0].symbol
    );
    assert_eq!(
        single.specializations[0].scratch,
        artifact.specializations[0].scratch
    );
    assert!(
        provider
            .compile_native(&[requests[0]; 2], &target)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    eprintln!(
        "{} mixed Reduce/Scan/Load specializations, {} LTO bytes",
        requests.len(),
        artifact.ltoir.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA_HOME and a Linux CUDA toolkit with NVRTC LTO support"]
fn native_mixed_scan_recipes() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = env::var("CUDA_HOME")?;
    let target = env::var("CUDA_COOP_TEST_TARGET").unwrap_or_else(|_| "compute_120".into());
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provider = Provider::from_checkout(&checkout, &toolkit)?;
    // Each operator is compiled in both modes; types/algorithms are spread
    // across those cases rather than multiplying independent dimensions.
    let types = [
        Dtype::I8,
        Dtype::U8,
        Dtype::I16,
        Dtype::U16,
        Dtype::I32,
        Dtype::U32,
        Dtype::I64,
        Dtype::U64,
        Dtype::F32,
        Dtype::F64,
        Dtype::I8,
        Dtype::I16,
        Dtype::U16,
        Dtype::U64,
    ];
    let operators = [
        ScanOperator::Sum,
        ScanOperator::Multiplies,
        ScanOperator::Min,
        ScanOperator::Max,
        ScanOperator::BitAnd,
        ScanOperator::BitOr,
        ScanOperator::BitXor,
    ];
    let algorithms = [
        ScanAlgorithm::Raking,
        ScanAlgorithm::RakingMemoize,
        ScanAlgorithm::WarpScans,
    ];
    let mut requests = Vec::new();
    for (index, dtype) in types.into_iter().enumerate() {
        let op = operators[index % operators.len()];
        let operation = if index < operators.len() {
            ScanOperation::Exclusive(op)
        } else {
            ScanOperation::Inclusive(op)
        };
        requests.push(NativeRequest::Scan(ScanRequest::block(
            dtype,
            operation,
            algorithms[index % algorithms.len()],
            [4, 4, 4],
            if index % 2 == 0 { 1 } else { 4 },
        )?));
    }
    requests.push(NativeRequest::Scan(ScanRequest::block(
        Dtype::I8,
        ScanOperation::ExclusiveSum,
        ScanAlgorithm::Raking,
        [8, 4, 1],
        9,
    )?));
    for (index, width) in [1, 2, 4, 8, 16, 32].into_iter().enumerate() {
        requests.push(NativeRequest::Scan(ScanRequest::warp(
            if index % 2 == 0 {
                Dtype::I8
            } else {
                Dtype::F64
            },
            match index % 3 {
                0 => ScanOperation::ExclusiveSum,
                1 => ScanOperation::Exclusive(ScanOperator::Min),
                _ => ScanOperation::Inclusive(ScanOperator::Max),
            },
            [4, 4, 4],
            width,
        )?));
    }
    for operation in [Operation::LoadFullValidity, Operation::StoreFullValidity] {
        requests.push(NativeRequest::LoadStore(Request::new_typed(
            Dtype::I32,
            operation,
            Algorithm::Direct,
            64,
            4,
        )?));
    }
    let artifact = provider.compile_native(&requests, &target)?;
    assert_eq!(artifact.specializations.len(), requests.len());
    for (request, entry) in requests.iter().zip(&artifact.specializations) {
        if matches!(request, NativeRequest::Scan(_)) {
            assert!(!entry.scratch_free);
            assert_eq!(entry.scratch, entry.cub_layout);
            assert!(entry.scratch.size > 0 && entry.scratch.alignment.is_power_of_two());
            assert!(!entry.mutates_payload, "Scan must preserve its input");
        } else {
            assert!(entry.scratch_free);
            assert_eq!(entry.scratch.size, 0);
        }
        eprintln!(
            "{request:?}: CUB {:?}, scratch {:?}",
            entry.cub_layout, entry.scratch
        );
    }
    let single = provider.compile_native(&requests[..1], &target)?;
    assert_eq!(
        single.specializations[0].identity,
        artifact.specializations[0].identity
    );
    assert_eq!(
        single.specializations[0].symbol,
        artifact.specializations[0].symbol
    );
    assert!(
        provider
            .compile_native(&[requests[0]; 2], &target)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    eprintln!(
        "{} mixed Scan/Load/Store specializations, {} LTO bytes",
        requests.len(),
        artifact.ltoir.len()
    );
    Ok(())
}

#[test]
fn rejects_invalid_shapes() {
    for (threads, items) in [(0, 1), (1025, 1), (128, 0), (1024, u32::MAX)] {
        assert!(Request::new(Operation::LoadFull, Algorithm::Direct, threads, items).is_err());
    }
    for algorithm in [Algorithm::WarpTranspose, Algorithm::WarpTransposeTimesliced] {
        assert!(
            Request::new(Operation::LoadFull, algorithm, 33, 4)
                .unwrap_err()
                .to_string()
                .contains("divisible by 32")
        );
        assert!(Request::new_with_shape(Operation::LoadFull, algorithm, [11, 3, 1], 4).is_err());
        assert!(Request::new_with_shape(Operation::LoadFull, algorithm, [8, 4, 2], 4).is_ok());
    }
    for block in [
        [0, 4, 4],
        [8, 0, 4],
        [8, 4, 0],
        [8, 4, 33],
        [u32::MAX, u32::MAX, 1],
    ] {
        assert!(Request::new_with_shape(Operation::LoadFull, Algorithm::Direct, block, 4).is_err());
    }
    assert!(
        Request::new_typed(Dtype::U64, Operation::LoadFull, Algorithm::Direct, 128, 4).is_err()
    );
    let warp = Request::new(Operation::LoadMaskedValidity, Algorithm::Direct, 64, 4).unwrap();
    for width in [0, 3, 6, 33, 64] {
        assert!(warp.with_warp_width(width).is_err());
    }
    for width in [1, 2, 4, 8, 16, 32] {
        assert!(warp.with_warp_width(width).is_ok());
        assert!(
            Request::new(Operation::LoadMaskedValidity, Algorithm::Direct, 48, 4)
                .unwrap()
                .with_warp_width(width)
                .is_err()
        );
    }
    for algorithm in [Algorithm::WarpTranspose, Algorithm::WarpTransposeTimesliced] {
        assert!(
            Request::new(Operation::LoadMaskedValidity, algorithm, 64, 4)
                .unwrap()
                .with_warp_width(8)
                .is_err()
        );
    }
    for algorithm in [
        Algorithm::Direct,
        Algorithm::Striped,
        Algorithm::Vectorize,
        Algorithm::Transpose,
    ] {
        assert!(
            Request::new(Operation::LoadMaskedValidity, algorithm, 64, 4)
                .unwrap()
                .with_warp_width(8)
                .is_ok()
        );
    }
    assert!(
        Request::new(Operation::LoadFull, Algorithm::Direct, 64, 4)
            .unwrap()
            .with_warp_width(8)
            .is_err()
    );
}

#[test]
#[ignore = "requires CUDA_HOME and a Linux CUDA toolkit with NVRTC LTO support"]
fn native_warp_algorithms_and_padded_masks() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = env::var("CUDA_HOME")?;
    let target = env::var("CUDA_COOP_TEST_TARGET").unwrap_or_else(|_| "compute_120".into());
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provider = Provider::from_checkout(&checkout, &toolkit)?;
    let mut cases = Vec::new();
    for width in [1, 8, 32] {
        for dtype in [Dtype::I8, Dtype::I32, Dtype::F64] {
            for items in [1, 4] {
                cases.push((width, dtype, items, Algorithm::Transpose));
            }
        }
    }
    for width in [2, 4, 16] {
        cases.push((width, Dtype::I32, 4, Algorithm::Transpose));
    }
    cases.push((8, Dtype::I8, 9, Algorithm::Transpose));
    for algorithm in [Algorithm::Striped, Algorithm::Vectorize] {
        for width in [1, 8] {
            cases.push((width, Dtype::I32, 4, algorithm));
        }
    }
    let mut requests = Vec::new();
    for &(width, dtype, items, algorithm) in &cases {
        for operation in [
            Operation::LoadFullValidity,
            Operation::LoadMaskedValidity,
            Operation::LoadFilledValidity,
            Operation::StoreFullValidity,
            Operation::StorePartialValidity,
        ] {
            requests.push(
                Request::new_typed_with_shape(dtype, operation, algorithm, [8, 4, 2], items)?
                    .with_warp_width(width)?,
            );
        }
    }
    let artifact = provider.compile(&requests, &target)?;
    for ((width, dtype, items, algorithm), entries) in cases
        .into_iter()
        .zip(artifact.specializations.chunks_exact(5))
    {
        for (index, entry) in entries.iter().enumerate() {
            eprintln!(
                "Warp{width} {dtype:?} {algorithm:?} N{items} operation{index}: CUB {:?}, total {:?}",
                entry.cub_layout, entry.scratch
            );
            if algorithm != Algorithm::Transpose {
                assert_eq!(
                    entry.scratch,
                    Layout {
                        size: 0,
                        alignment: 1
                    }
                );
                assert!(!entry.mutates_payload);
            } else {
                assert_eq!(entry.mutates_payload, index >= 3);
                assert!(entry.scratch.alignment >= entry.cub_layout.alignment);
                if index == 4 && items > 1 {
                    assert!(entry.scratch.alignment >= 4);
                    assert!(entry.scratch.size >= entry.cub_layout.size + 4 * items as usize);
                    if (width, dtype, items) == (8, Dtype::I8, 4) {
                        // CUB has 33 byte-aligned cells; the actual combined C++
                        // struct pads before its 16-byte unsigned-int mask plane.
                        assert_eq!(
                            entry.cub_layout,
                            Layout {
                                size: 33,
                                alignment: 1
                            }
                        );
                        assert_eq!(
                            entry.scratch,
                            Layout {
                                size: 52,
                                alignment: 4
                            }
                        );
                    }
                } else {
                    assert_eq!(entry.scratch, entry.cub_layout);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn preserves_exact_shape_and_legacy_constructors() {
    let old = Request::new_typed(
        Dtype::U8,
        Operation::LoadFullValidity,
        Algorithm::Direct,
        128,
        4,
    )
    .unwrap();
    let shaped = Request::new_typed_with_shape(
        Dtype::U8,
        Operation::LoadFullValidity,
        Algorithm::Direct,
        [128, 1, 1],
        4,
    )
    .unwrap();
    assert_eq!(old, shaped);
    assert_eq!(shaped.block_shape(), [128, 1, 1]);
    let three = Request::new_typed_with_shape(
        Dtype::U8,
        Operation::LoadFullValidity,
        Algorithm::Direct,
        [8, 4, 4],
        4,
    )
    .unwrap();
    assert_eq!(three.block_threads(), old.block_threads());
    assert_ne!(three, old);
    assert_eq!(
        Request::new(Operation::LoadFull, Algorithm::Direct, 128, 4).unwrap(),
        Request::new_with_shape(Operation::LoadFull, Algorithm::Direct, [128, 1, 1], 4).unwrap(),
    );
}

#[test]
#[ignore = "requires CUDA_HOME and a Linux CUDA toolkit with NVRTC LTO support"]
fn native_typed_validity_and_determinism() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = env::var("CUDA_HOME")?;
    let target = env::var("CUDA_COOP_TEST_TARGET").unwrap_or_else(|_| "compute_120".into());
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provider = Provider::from_checkout(&checkout, &toolkit)?;
    let mut requests = Vec::new();
    for dtype in [
        Dtype::I8,
        Dtype::U8,
        Dtype::I16,
        Dtype::U16,
        Dtype::I32,
        Dtype::U32,
        Dtype::I64,
        Dtype::U64,
        Dtype::F32,
        Dtype::F64,
    ] {
        for (algorithm, threads) in [
            (Algorithm::Direct, 128),
            (Algorithm::Striped, 33),
            (Algorithm::Vectorize, 33),
            (Algorithm::Transpose, 33),
            (Algorithm::WarpTranspose, 64),
            (Algorithm::WarpTransposeTimesliced, 64),
        ] {
            for items in [1, 4] {
                for operation in [
                    Operation::LoadFullValidity,
                    Operation::LoadMaskedValidity,
                    Operation::LoadFilledValidity,
                    Operation::StoreFullValidity,
                    Operation::StorePartialValidity,
                ] {
                    requests.push(Request::new_typed(
                        dtype, operation, algorithm, threads, items,
                    )?);
                }
            }
        }
    }
    // The same five typed Direct recipes support each canonical logical width
    // inside a multidimensional block; their actual queried scratch remains zero.
    for width in [1, 2, 4, 8, 16, 32] {
        for operation in [
            Operation::LoadFullValidity,
            Operation::LoadMaskedValidity,
            Operation::LoadFilledValidity,
            Operation::StoreFullValidity,
            Operation::StorePartialValidity,
        ] {
            requests.push(
                Request::new_with_shape(operation, Algorithm::Direct, [8, 4, 2], 4)?
                    .with_warp_width(width)?,
            );
        }
    }
    // Partial warps, complete warps, maximum block size, CUB bank padding and
    // packed-mask byte boundaries exercise different actual storage layouts.
    for (threads, items) in [
        (1, 4),
        (31, 4),
        (32, 4),
        (128, 4),
        (130, 4),
        (1024, 1),
        (33, 8),
        (33, 9),
    ] {
        for operation in [
            Operation::LoadMaskedValidity,
            Operation::StoreFullValidity,
            Operation::StorePartialValidity,
        ] {
            requests.push(Request::new(
                operation,
                Algorithm::Transpose,
                threads,
                items,
            )?);
        }
    }
    // The first shape has a partial physical warp crossing rows; the second
    // spans four Z planes, one warp each, and matches an existing 1D total.
    for block in [[11, 3, 1], [8, 4, 4]] {
        for algorithm in [
            Algorithm::Direct,
            Algorithm::Striped,
            Algorithm::Vectorize,
            Algorithm::Transpose,
        ] {
            for items in [1, 4] {
                for operation in [
                    Operation::LoadFullValidity,
                    Operation::LoadMaskedValidity,
                    Operation::LoadFilledValidity,
                    Operation::StoreFullValidity,
                    Operation::StorePartialValidity,
                ] {
                    requests.push(Request::new_with_shape(operation, algorithm, block, items)?);
                }
            }
        }
    }
    // Odd item counts retain CUB's scalar fallback with the same validity ABI.
    for operation in [Operation::LoadFullValidity, Operation::StoreFullValidity] {
        requests.push(Request::new(operation, Algorithm::Vectorize, 33, 3)?);
    }
    for algorithm in [Algorithm::WarpTranspose, Algorithm::WarpTransposeTimesliced] {
        for items in [1, 4] {
            for operation in [
                Operation::LoadFullValidity,
                Operation::LoadMaskedValidity,
                Operation::LoadFilledValidity,
                Operation::StoreFullValidity,
                Operation::StorePartialValidity,
            ] {
                requests.push(Request::new_with_shape(
                    operation,
                    algorithm,
                    [8, 4, 2],
                    items,
                )?);
            }
        }
    }
    let first = provider.compile(&requests, &target)?;
    let repeated = provider.compile(&requests, &target)?;
    assert_eq!(first.ltoir, repeated.ltoir);
    assert_eq!(first.identity, repeated.identity);
    assert_eq!(first.specializations.len(), requests.len());
    let mut identities = std::collections::HashSet::new();
    for (entry, request) in first.specializations.iter().zip(&requests) {
        assert!(identities.insert(entry.identity));
        eprintln!(
            "{request:?}: CUB {:?}, scratch {:?}",
            entry.cub_layout, entry.scratch
        );
        if matches!(
            request.algorithm(),
            Algorithm::Direct | Algorithm::Striped | Algorithm::Vectorize
        ) {
            assert_eq!(
                entry.cub_layout,
                Layout {
                    size: 1,
                    alignment: 1
                }
            );
            assert_eq!(
                entry.scratch,
                Layout {
                    size: 0,
                    alignment: 1
                }
            );
            assert!(entry.scratch_free);
            assert!(!entry.mutates_payload);
        } else {
            assert!(!entry.scratch_free);
            assert_eq!(
                entry.mutates_payload,
                matches!(
                    request.operation(),
                    Operation::StoreFullValidity | Operation::StorePartialValidity
                )
            );
            assert_eq!(entry.scratch.alignment, entry.cub_layout.alignment);
            if request.dtype() == Dtype::I32 && request.block_threads() == 64 {
                let (load, store) = match (request.algorithm(), request.items_per_thread()) {
                    (Algorithm::WarpTranspose, 1) => (256, 272),
                    (Algorithm::WarpTranspose, 4) => (1024, 1040),
                    (Algorithm::WarpTransposeTimesliced, 1) => (128, 144),
                    (Algorithm::WarpTransposeTimesliced, 4) => (512, 528),
                    _ => unreachable!(),
                };
                let is_store = matches!(
                    request.operation(),
                    Operation::StoreFullValidity | Operation::StorePartialValidity
                );
                assert_eq!(
                    entry.cub_layout,
                    Layout {
                        size: if is_store { store } else { load },
                        alignment: 16
                    }
                );
            }
            if request.operation() == Operation::StorePartialValidity
                && request.items_per_thread() > 1
            {
                let plane = 4
                    * request.block_threads().div_ceil(32) as usize
                    * request.items_per_thread() as usize;
                let aligned_total = (entry.cub_layout.size + plane)
                    .div_ceil(entry.cub_layout.alignment)
                    * entry.cub_layout.alignment;
                assert_eq!(entry.scratch.size, aligned_total);
                assert!(entry.scratch.size > entry.cub_layout.size);
                if request.block_threads() == 128 && request.items_per_thread() == 4 {
                    assert_eq!(entry.cub_layout.size, 2064);
                    assert_eq!(entry.scratch.size, 2128);
                }
            } else {
                assert_eq!(entry.scratch, entry.cub_layout);
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA_HOME and a Linux CUDA toolkit with NVRTC LTO support"]
fn native_specializations_and_determinism() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = env::var("CUDA_HOME")?;
    let target = env::var("CUDA_COOP_TEST_TARGET").unwrap_or_else(|_| "compute_120".into());
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let start = Instant::now();
    let provider = Provider::from_checkout(&checkout, &toolkit)?;
    eprintln!("provider initialization: {:?}", start.elapsed());
    let mut requests = Vec::new();
    for algorithm in [
        Algorithm::Direct,
        Algorithm::Striped,
        Algorithm::Vectorize,
        Algorithm::Transpose,
        Algorithm::WarpTranspose,
        Algorithm::WarpTransposeTimesliced,
    ] {
        for items in [1, 4] {
            for operation in [
                Operation::LoadFull,
                Operation::LoadFilled,
                Operation::StoreFull,
                Operation::StorePartial,
            ] {
                requests.push(Request::new(operation, algorithm, 128, items)?);
            }
        }
    }
    for algorithm in [
        Algorithm::Direct,
        Algorithm::Striped,
        Algorithm::Vectorize,
        Algorithm::Transpose,
        Algorithm::WarpTranspose,
        Algorithm::WarpTransposeTimesliced,
    ] {
        for operation in [
            Operation::LoadFull,
            Operation::LoadFilled,
            Operation::StoreFull,
            Operation::StorePartial,
        ] {
            requests.push(Request::new_with_shape(operation, algorithm, [8, 4, 4], 4)?);
        }
    }
    let start = Instant::now();
    let first = provider.compile(&requests, &target)?;
    eprintln!(
        "{}-specialization batch: {:?}, {} bytes",
        requests.len(),
        start.elapsed(),
        first.ltoir.len()
    );
    let second = provider.compile(&requests, &target)?;
    assert_eq!(first.identity, second.identity);
    assert_eq!(first.ltoir, second.ltoir, "repeated native output changed");
    assert_eq!(first.specializations.len(), requests.len());
    let mut identities = std::collections::HashSet::new();
    for ((entry, repeated), request) in first
        .specializations
        .iter()
        .zip(&second.specializations)
        .zip(&requests)
    {
        assert_eq!(entry.identity, repeated.identity);
        assert_eq!(entry.symbol, repeated.symbol);
        assert_eq!(
            entry.mutates_payload,
            !matches!(
                request.algorithm(),
                Algorithm::Direct | Algorithm::Striped | Algorithm::Vectorize
            ) && matches!(
                request.operation(),
                Operation::StoreFull | Operation::StorePartial
            )
        );
        assert!(identities.insert(entry.identity));
        eprintln!(
            "{request:?}: CUB {:?}, scratch {:?}",
            entry.cub_layout, entry.scratch
        );
        if matches!(
            request.algorithm(),
            Algorithm::Direct | Algorithm::Striped | Algorithm::Vectorize
        ) {
            assert!(entry.scratch_free);
            assert_eq!(
                entry.scratch,
                Layout {
                    size: 0,
                    alignment: 1
                }
            );
            assert_eq!(
                entry.cub_layout,
                Layout {
                    size: 1,
                    alignment: 1
                }
            );
        } else {
            assert!(!entry.scratch_free);
            assert_eq!(entry.scratch, entry.cub_layout);
            let exchange_threads = if request.algorithm() == Algorithm::WarpTransposeTimesliced {
                32
            } else {
                128
            };
            assert!(
                entry.scratch.size >= exchange_threads * request.items_per_thread() as usize * 4
            );
        }
    }
    // A specialization's symbol is independent of unrelated batch members.
    let single = provider.compile(&requests[..1], &target)?;
    assert_eq!(
        single.specializations[0].identity,
        first.specializations[0].identity
    );
    assert_eq!(
        single.specializations[0].symbol,
        first.specializations[0].symbol
    );
    assert_ne!(single.identity, first.identity);
    assert!(
        provider
            .compile(&[], &target)
            .unwrap_err()
            .to_string()
            .contains("at least one")
    );
    assert!(
        provider
            .compile(&[requests[0]; 2], &target)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    assert!(
        provider
            .compile(&requests[..1], "compute_120;injected")
            .unwrap_err()
            .to_string()
            .contains("target must")
    );
    assert!(
        provider
            .compile(&requests[..1], "compute_999999")
            .unwrap_err()
            .to_string()
            .contains("nvrtcCompileProgram")
    );
    Ok(())
}
