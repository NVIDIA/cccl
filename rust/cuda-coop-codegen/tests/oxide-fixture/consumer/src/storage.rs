// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_device::{CollectiveTiles, DisjointSlice, kernel, launch_contract};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy<const ITEMS: usize, const TRANSPOSE: bool, const MODE: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
    ) {
        // SAFETY: host-owned buffers cover every canonical tile; root geometry
        // and uniform controls satisfy the private wrapper's stated contract.
        unsafe {
            coop_marker_fixture_helper::storage::roundtrip::<ITEMS, TRANSPOSE, MODE>(
                input,
                &mut output,
                valid,
            );
        }
    }
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    let image = std::fs::read(std::env::current_exe()?)?;
    let bundles = oxide_artifacts::read_artifact_bundles_from_object_bytes(&image)?;
    let bundle = bundles
        .iter()
        .find(|bundle| {
            bundle
                .payload(oxide_artifacts::ArtifactPayloadKind::Cubin)
                .is_some()
        })
        .ok_or("missing native storage artifact")?;
    std::fs::write(
        "storage.cubin",
        bundle
            .payload(oxide_artifacts::ArtifactPayloadKind::Cubin)
            .unwrap(),
    )?;
    for entry in &bundle.entries {
        if let Some(facts) = entry.launch_facts {
            println!("ENTRY {} {:?}", entry.symbol, facts);
        }
    }
    if std::env::var_os("OXIDE_STORAGE_INSPECT_ONLY").is_some() {
        return Ok(());
    }
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |c| c.clone())?;
    let stream = context.default_stream();
    let module = unsafe { kernels::load_async(0)? };
    let input: Vec<_> = (0..1536).map(|i| i * 3 + 17).collect();
    let mut cases = 0;
    macro_rules! check {
        ($i:literal,$transpose:literal,$mode:literal) => {{
            let plan = module
                .prepare_copy::<$i, $transpose, $mode>(LaunchConfig1D::auto_shared(3, 128))?;
            println!(
                "storage I={} transpose={} mode={} bytes={}",
                $i,
                $transpose,
                $mode,
                plan.__raw_config().shared_mem_bytes
            );
            assert!(
                module
                    .prepare_copy::<$i, $transpose, $mode>(LaunchConfig1D::auto_shared(3, 64))
                    .is_err()
            );
            let single = if $i == 1 { 528 } else { 2064 };
            let expected = if !$transpose {
                0
            } else if $mode == 3 {
                single + 2 * 512 * $i
            } else if $mode == 4 {
                single + 512 * $i
            } else if $mode == 6 {
                1536 + single + 512 * $i
            } else if $mode == 7 {
                ((64 + single + 63) & !63) + 512 * $i
            } else {
                single
            };
            let actual = plan.__raw_config().shared_mem_bytes;
            if $mode == 7 {
                // Either descriptor may be placed first after the native
                // prefix. Both orders obey the same alignment requirements.
                assert!(actual == expected || actual == 64 + 512 * $i + single);
            } else {
                assert_eq!(actual, expected);
            }
            if actual > 0 {
                assert!(
                    module
                        .prepare_copy::<$i, $transpose, $mode>(LaunchConfig1D::new(
                            3,
                            128,
                            actual - 1
                        ))
                        .is_err()
                );
            }
            for valid in [0, 1, 128 * $i - 1, 128 * $i] {
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let target = DeviceBuffer::from_host(&stream, &vec![-99; input.len()])?;
                let (_source, target) = module
                    .copy_async_owned::<$i, $transpose, $mode, _, _>(&plan, source, target, valid)
                    .sync()?;
                let mut result = vec![0; input.len()];
                target.copy_to_host(&stream, &mut result)?;
                stream.synchronize()?;
                for (index, value) in result.into_iter().enumerate() {
                    assert_eq!(
                        value,
                        if index % 512 < valid {
                            input[index]
                                + if $mode == 5 {
                                    ((index % 512 / $i + 1) % 128) as i32
                                } else if $mode == 6 {
                                    (((index % 512 / $i + 1) % 128) * 3 + 1) as i32
                                } else if $mode == 7 {
                                    (index % 512 / $i % 8 + 10) as i32
                                } else if $mode == 8 {
                                    (1 + (valid & 3)) as i32
                                } else {
                                    0
                                }
                        } else {
                            -99
                        },
                        "I={} transpose={} mode={} valid={valid} index={index}",
                        $i,
                        $transpose,
                        $mode
                    );
                }
                cases += 1;
            }
        }};
    }
    check!(1, false, 0);
    check!(4, false, 0);
    check!(1, true, 0);
    check!(4, true, 0);
    check!(1, true, 1);
    check!(4, true, 1);
    check!(1, true, 2);
    check!(4, true, 2);
    check!(1, true, 3);
    check!(4, true, 3);
    check!(1, true, 4);
    check!(4, true, 4);
    check!(1, true, 5);
    check!(4, true, 5);
    check!(1, true, 6);
    check!(4, true, 6);
    check!(1, true, 7);
    check!(4, true, 7);
    check!(1, true, 8);
    check!(4, true, 8);
    check!(1, true, 9);
    check!(4, true, 9);
    check!(1, false, 2);
    check!(4, false, 2);
    println!("PASS: {cases} storage cases with three blocks and boundary counts");
    Ok(())
}
