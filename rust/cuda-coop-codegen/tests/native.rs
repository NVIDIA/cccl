// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Real native compiler coverage; ordinary host tests need no CUDA installation.

use cuda_coop_codegen::{Algorithm, Dtype, Layout, Operation, Provider, Request};
use std::{env, path::Path, time::Instant};

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
