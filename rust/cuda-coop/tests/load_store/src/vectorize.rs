// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Vectorize full/guarded access and post-Store validity, including misaligned offsets.
use cuda_coop as coop;
use cuda_device::{CollectiveTiles, DisjointSlice, kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    fn exercise<T: coop::Numeric + PartialEq, const ITEMS: usize>(
        input: &[T],
        output: &mut DisjointSlice<T, CollectiveTiles<8>>,
        valid: usize,
        shift: usize,
        default: T,
        mode: u32,
    ) {
        let block = coop::this_block();
        let x = thread::blockDim_x() as usize;
        let y = thread::blockDim_y() as usize;
        let threads = x * y * thread::blockDim_z() as usize;
        let lane = thread::threadIdx_x() as usize
            + x * (thread::threadIdx_y() as usize + y * thread::threadIdx_z() as usize);
        let group = thread::blockIdx_x() as usize
            + thread::gridDim_x() as usize
                * (thread::blockIdx_y() as usize
                    + thread::gridDim_y() as usize * thread::blockIdx_z() as usize);
        let offset = group * threads * 8 + shift;
        let mut values = coop::ThreadData::<_, ITEMS>::new();
        match mode {
            0 => coop::load(
                block,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .algorithm(coop::Vectorize)
                    .offset(offset),
            ),
            1 | 5 | 6 | 9 => {
                // A second masked Load must clear validity left by a full Load.
                coop::load(
                    block,
                    input,
                    &mut values,
                    coop::LoadOptions::default()
                        .algorithm(coop::Vectorize)
                        .offset(offset),
                );
                coop::load(
                    block,
                    input,
                    &mut values,
                    coop::LoadOptions::default()
                        .algorithm(coop::Vectorize)
                        .offset(offset)
                        .valid_items(valid),
                );
            }
            2 | 4 => coop::load(
                block,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .algorithm(coop::Vectorize)
                    .offset(offset)
                    .valid_items(valid)
                    .oob_default(default),
            ),
            7 => {}
            _ => {
                let mut item = 0;
                while item < ITEMS {
                    let selected = lane * ITEMS + item < valid;
                    if (selected || (lane + item) & 1 == 0)
                        && !(mode == 8 && lane * ITEMS + item == valid - 1)
                    {
                        values.write(item, default);
                    }
                    item += 1;
                }
            }
        }
        let before = values;
        if mode == 0 || mode == 2 || mode == 9 {
            coop::store(
                block,
                &mut *output,
                &values,
                coop::StoreOptions::default()
                    .algorithm(coop::Vectorize)
                    .offset(offset),
            );
        } else {
            let written = if mode == 5 {
                valid / 2
            } else if mode == 6 {
                valid + 1
            } else {
                valid
            };
            coop::store(
                block,
                &mut *output,
                &values,
                coop::StoreOptions::default()
                    .algorithm(coop::Vectorize)
                    .offset(offset)
                    .valid_items(written),
            );
        }
        let mut observations_ok = true;
        let mut item = 0;
        while item < ITEMS {
            let position = lane * ITEMS + item;
            let expected = match mode {
                3 | 8 => {
                    if position < valid || (lane + item) & 1 == 0 {
                        Some(default)
                    } else {
                        None
                    }
                }
                7 => None,
                _ => {
                    if position < valid || mode == 0 {
                        Some(input[offset + position])
                    } else if mode == 2 || mode == 4 {
                        Some(default)
                    } else {
                        None
                    }
                }
            };
            observations_ok &= before.get(item) == expected;
            observations_ok &= values.get(item) == before.get(item);
            item += 1;
        }
        if mode == 3 || mode == 4 {
            thread::sync_threads();
            // A repeated zero Store preserves arbitrary initialized/invalid slots.
            coop::store(
                block,
                &mut *output,
                &values,
                coop::StoreOptions::default()
                    .algorithm(coop::Vectorize)
                    .offset(offset)
                    .valid_items(0),
            );
            let mut item = 0;
            while item < ITEMS {
                observations_ok &= values.get(item) == before.get(item);
                item += 1;
            }
        }
        assert!(observations_ok);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn partial_33<T: coop::Numeric + PartialEq, const ITEMS: usize>(
        input: &[T],
        mut output: DisjointSlice<T, CollectiveTiles<8>>,
        valid: usize,
        shift: usize,
        default: T,
        mode: u32,
    ) {
        exercise::<_, ITEMS>(input, &mut output, valid, shift, default, mode);
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (8, 4, 4), dynamic_shared = inferred)]
    pub fn partial_3d<T: coop::Numeric + PartialEq, const ITEMS: usize>(
        input: &[T],
        mut output: DisjointSlice<T, CollectiveTiles<8>>,
        valid: usize,
        shift: usize,
        default: T,
        mode: u32,
    ) {
        exercise::<_, ITEMS>(input, &mut output, valid, shift, default, mode);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn vectorize_scalar(mut output: DisjointSlice<i32, CollectiveTiles<1>>, valid: usize) {
        let offset = thread::blockIdx_x() as usize * 33;
        coop::store(
            coop::this_block(),
            &mut output,
            17_i32,
            coop::StoreOptions::default()
                .algorithm(coop::Vectorize)
                .offset(offset)
                .valid_items(valid),
        );
    }
}

pub fn run(trap: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig3D};
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: embedded kernels and typed declarations compile together.
    let module = unsafe { kernels::load_async(0)? };
    if let Some(trap) = trap {
        let mode = match trap {
            "vectorize-larger" => 6,
            "vectorize-full-hole" => 9,
            _ => 8,
        };
        let plan = module.prepare_partial_33::<i32, 4>(LaunchConfig1D::auto_shared(1, 33))?;
        let source = DeviceBuffer::from_host(&stream, &vec![7_i32; 132])?;
        let target = DeviceBuffer::from_host(&stream, &vec![0_i32; 132])?;
        module
            .partial_33_async_owned::<i32, 4, _, _>(&plan, source, target, 34, 0, -7, mode)
            .sync()?;
        return Err("invalid Vectorize Store unexpectedly completed".into());
    }
    let mut cases = 0;
    macro_rules! run {
        ($t:ty, $items:literal, $threads:literal, $prepare:ident, $launch:ident, $config:expr, $extra:expr) => {{
            let tile = $threads * $items;
            let plan = module.$prepare::<$t, $items>($config)?;
            assert_eq!(plan.__raw_config().shared_mem_bytes, 0);
            let prefix = ($threads + 1).min(tile);
            let mut selected = vec![
                (0, tile, 0),
                (0, tile, 1),
                (0, tile, 4),
                (1, prefix, 0),
                (2, 1, 0),
                (3, prefix, 0),
                (5, prefix, 0),
            ];
            if $extra {
                for valid in [0, 1, $threads - 1, $threads, prefix, tile] {
                    for mode in [1, 2, 3, 4, 5] {
                        selected.push((mode, valid, 0));
                    }
                }
                for mode in [1, 3, 4, 5] {
                    selected.push((mode, tile - 7, 7));
                }
                selected.push((7, 0, 0));
            }
            selected.sort_unstable();
            selected.dedup();
            for (mode, valid, shift) in selected {
                let written = if mode == 0 || mode == 2 || mode == 9 {
                    tile
                } else if mode == 5 {
                    valid / 2
                } else {
                    valid
                };
                // Keep the final allocation shorter than the full tile when
                // counted Store writes a prefix. Its guarded iterator must not
                // form an out-of-allocation pointer.
                let output_len = $threads * 8 + shift + written;
                let input: Vec<$t> = (0..2 * $threads * 8 + shift)
                    .map(|i| (i % 101) as $t)
                    .collect();
                let default = 111 as $t;
                let sentinel = 123 as $t;
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let target = DeviceBuffer::from_host(&stream, &vec![sentinel; output_len])?;
                let (_, target) = module
                    .$launch::<$t, $items, _, _>(&plan, source, target, valid, shift, default, mode)
                    .sync()?;
                let mut output = vec![sentinel; output_len];
                target.copy_to_host(&stream, &mut output)?;
                stream.synchronize()?;
                for (index, actual) in output.into_iter().enumerate() {
                    let local = index % ($threads * 8);
                    let expected = if local >= shift && local - shift < written {
                        if mode == 3 || (mode == 2 && local - shift >= valid) {
                            default
                        } else {
                            input[index]
                        }
                    } else {
                        sentinel
                    };
                    assert_eq!(
                        actual,
                        expected,
                        "type={} items={} mode={mode} valid={valid} shift={shift} index={index}",
                        stringify!($t),
                        $items
                    );
                }
                cases += 1;
            }
        }};
    }
    macro_rules! width {
        ($t:ty) => {
            run!(
                $t,
                1,
                33,
                prepare_partial_33,
                partial_33_async_owned,
                LaunchConfig1D::auto_shared(2, 33),
                false
            );
            run!(
                $t,
                4,
                33,
                prepare_partial_33,
                partial_33_async_owned,
                LaunchConfig1D::auto_shared(2, 33),
                false
            );
        };
    }
    width!(i8);
    width!(u8);
    width!(i16);
    width!(u16);
    width!(i32);
    width!(u32);
    width!(i64);
    width!(u64);
    width!(f32);
    width!(f64);
    run!(
        i32,
        4,
        33,
        prepare_partial_33,
        partial_33_async_owned,
        LaunchConfig1D::auto_shared(2, 33),
        true
    );
    run!(
        i32,
        4,
        128,
        prepare_partial_3d,
        partial_3d_async_owned,
        LaunchConfig3D::auto_shared((1, 2, 1), (8, 4, 4)),
        true
    );
    run!(
        i32,
        3,
        33,
        prepare_partial_33,
        partial_33_async_owned,
        LaunchConfig1D::auto_shared(2, 33),
        false
    );
    for valid in [0, 1, 32, 33] {
        let plan = module.prepare_vectorize_scalar(LaunchConfig1D::auto_shared(2, 33))?;
        assert_eq!(plan.__raw_config().shared_mem_bytes, 0);
        let target = DeviceBuffer::from_host(&stream, &vec![-1_i32; 33 + valid])?;
        let target = module
            .vectorize_scalar_async_owned(&plan, target, valid)
            .sync()?;
        let mut output = vec![0_i32; 33 + valid];
        target.copy_to_host(&stream, &mut output)?;
        stream.synchronize()?;
        for (i, value) in output.into_iter().enumerate() {
            assert_eq!(value, if i % 33 < valid { 17 } else { -1 });
        }
        cases += 1;
    }
    println!("PASS: {cases} Vectorize cases with observed payload mapping and validity");
    Ok(())
}
