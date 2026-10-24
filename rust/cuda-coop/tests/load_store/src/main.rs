// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use coop::{Numeric, ThreadData, this_block};
mod striped;
mod transpose;

use cuda_coop as coop;
use cuda_device::{CollectiveTiles, DisjointSlice, kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn exercise<T: Numeric + PartialEq, const ITEMS_PER_THREAD: usize>(
        input: &[T],
        output: &mut DisjointSlice<T, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: T,
        mode: u32,
    ) {
        let block = this_block();
        let x = thread::blockDim_x() as usize;
        let y = thread::blockDim_y() as usize;
        let threads = x * y * thread::blockDim_z() as usize;
        let lane = thread::threadIdx_x() as usize
            + x * (thread::threadIdx_y() as usize + y * thread::threadIdx_z() as usize);
        let group = thread::blockIdx_x() as usize
            + thread::gridDim_x() as usize
                * (thread::blockIdx_y() as usize
                    + thread::gridDim_y() as usize * thread::blockIdx_z() as usize);
        let offset = group * threads * 4 + shift;
        let mut values = ThreadData::<_, ITEMS_PER_THREAD>::new();
        match mode {
            0 => coop::load(
                block,
                input,
                &mut values,
                coop::LoadOptions::default().offset(offset),
            ),
            1 | 5 | 6 => {
                coop::load(
                    block,
                    input,
                    &mut values,
                    coop::LoadOptions::default().offset(offset),
                );
                coop::load(
                    block,
                    input,
                    &mut values,
                    coop::LoadOptions::default()
                        .offset(offset)
                        .valid_items(valid),
                );
            }
            2 | 4 => coop::load(
                block,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .valid_items(valid)
                    .oob_default(default),
            ),
            9 => {}
            10 => coop::load(
                block,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .oob_default(default),
            ),
            11 => {
                coop::load(
                    block,
                    input,
                    &mut values,
                    coop::LoadOptions::default().offset(offset).valid_items(0),
                );
                let _invalid = values[0];
            }
            _ => {
                // Real nonadjacent initialized slots beyond the stored prefix.
                // The two masks differ, then reconverged lane-local selection
                // moves each value together with its validity metadata.
                let mut other = ThreadData::<_, ITEMS_PER_THREAD>::new();
                let mut item = 0;
                while item < ITEMS_PER_THREAD {
                    let prefix = lane * ITEMS_PER_THREAD + item < valid;
                    if prefix || item & 1 == 1 {
                        values.write(item, default);
                    }
                    if prefix || item & 1 == 0 {
                        other.write(item, default);
                    }
                    item += 1;
                }
                values = if lane & 1 == 0 { values } else { other };
            }
        }
        let before = values;
        if mode == 0 || mode == 2 {
            coop::store(
                block,
                &mut *output,
                &values,
                coop::StoreOptions::default().offset(offset),
            );
        } else {
            let written = if mode == 5 {
                valid + 1
            } else if mode == 6 {
                valid / 2
            } else {
                valid
            };
            coop::store(
                block,
                &mut *output,
                &values,
                coop::StoreOptions::default()
                    .offset(offset)
                    .valid_items(written),
            );
        }
        let mut observations_ok = true;
        let mut item = 0;
        while item < ITEMS_PER_THREAD {
            if let Some(value) = before.get(item) {
                observations_ok &= values.get(item) == Some(value);
            } else {
                observations_ok &= values.get(item).is_none();
            }
            if mode == 1 || mode == 6 {
                observations_ok &=
                    values.get(item).is_some() == (lane * ITEMS_PER_THREAD + item < valid);
            }
            item += 1;
        }
        if mode == 4 {
            thread::sync_threads();
            coop::load(
                block,
                &mut *output,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .valid_items(valid)
                    .oob_default(default),
            );
            thread::sync_threads();
            coop::store(
                block,
                &mut *output,
                &values,
                coop::StoreOptions::default()
                    .offset(offset)
                    .valid_items(valid),
            );
        }
        assert!(observations_ok);
        // Final writes complete through the owned async host operation.
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_i8_128<const ITEMS_PER_THREAD: usize>(
        input: &[i8],
        mut output: DisjointSlice<i8, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: i8,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_u8_128<const ITEMS_PER_THREAD: usize>(
        input: &[u8],
        mut output: DisjointSlice<u8, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: u8,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_i16_128<const ITEMS_PER_THREAD: usize>(
        input: &[i16],
        mut output: DisjointSlice<i16, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: i16,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_u16_128<const ITEMS_PER_THREAD: usize>(
        input: &[u16],
        mut output: DisjointSlice<u16, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: u16,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_i32_128<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: i32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_u32_128<const ITEMS_PER_THREAD: usize>(
        input: &[u32],
        mut output: DisjointSlice<u32, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: u32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_i64_128<const ITEMS_PER_THREAD: usize>(
        input: &[i64],
        mut output: DisjointSlice<i64, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: i64,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_u64_128<const ITEMS_PER_THREAD: usize>(
        input: &[u64],
        mut output: DisjointSlice<u64, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: u64,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_f32_128<const ITEMS_PER_THREAD: usize>(
        input: &[f32],
        mut output: DisjointSlice<f32, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: f32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_f64_128<const ITEMS_PER_THREAD: usize>(
        input: &[f64],
        mut output: DisjointSlice<f64, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: f64,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn copy_i32_64<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: i32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }
    #[kernel]
    #[launch_contract(domain = 2, block = (11, 3, 1), dynamic_shared = inferred)]
    pub fn copy_i32_2d<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: i32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }
    #[kernel]
    #[launch_contract(domain = 3, block = (8, 4, 4), dynamic_shared = inferred)]
    pub fn copy_i32_3d<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        default: i32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD>(input, &mut output, valid, shift, default, mode);
    }
    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn scalar(mut output: DisjointSlice<i16, CollectiveTiles<1>>, value: i16) {
        let block = this_block();
        let offset = block.rank_at::<coop::Grid>() as usize * block.count() as usize;
        coop::store(
            block,
            &mut output,
            value,
            coop::StoreOptions::default().offset(offset),
        );
        thread::sync_threads();
        let mut payload = ThreadData::<_, 1>::new();
        coop::load(
            block,
            &mut output,
            &mut payload,
            coop::LoadOptions::default().offset(offset),
        );
        let scalar_preserved = payload.get(0) == Some(value);
        thread::sync_threads();
        coop::load(
            block,
            &mut output,
            &mut payload,
            coop::LoadOptions::default().offset(offset).valid_items(1),
        );
        assert!(scalar_preserved && payload.get(0).is_some() == (thread::threadIdx_x() == 0));
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Some(argument) = std::env::args().nth(1) {
        if argument == "striped" {
            return striped::run(None);
        }
        if argument.starts_with("striped-") {
            return striped::run(Some(&argument));
        }
        if argument == "transpose" {
            return transpose::run(None);
        }
        if argument.starts_with("transpose-") {
            return transpose::run(Some(&argument));
        }
    }

    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig2D, LaunchConfig3D};
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: embedded kernels and these typed declarations are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    let trap = std::env::args().nth(1);
    let mut cases = 0;
    macro_rules! run {
        ($t:ty, $prepare:ident, $launch:ident, $block:literal, $items:literal, $default:expr) => {
            run!(
                $t,
                $prepare,
                $launch,
                $block,
                $items,
                $default,
                |groups| LaunchConfig1D::auto_shared(groups, $block),
                |groups| LaunchConfig1D::auto_shared(groups, $block / 2)
            );
        };
        ($t:ty, $prepare:ident, $launch:ident, $block:literal, $items:literal, $default:expr,
         $config:expr, $wrong_shape:expr) => {{
            let default: $t = $default;
            for groups in [1_u32, 3] {
                let prepared = module.$prepare::<$items>(($config)(groups))?;
                assert_eq!(prepared.__raw_config().shared_mem_bytes, 0);
                let error = module
                    .$prepare::<$items>(($wrong_shape)(groups))
                    .err()
                    .expect("a different exact block shape must reject");
                assert!(error.to_string().contains("does not match required block"));
                let (gx, gy, gz) = prepared.__raw_config().grid_dim;
                let groups = gx * gy * gz;
                for mode in [0, 1, 2, 3, 4, 6] {
                    for valid in [0, 1, $block * $items - 1, $block * $items] {
                        if (mode == 0) && valid != $block * $items {
                            continue;
                        }
                        let shift = if $items == 1 { 7 } else { 0 };
                        let end = (groups as usize - 1) * $block * 4 + shift;
                        let source_len = end
                            + if mode <= 1 || mode == 6 {
                                $block * $items
                            } else {
                                valid
                            };
                        let target_len = end + if mode <= 2 { $block * $items } else { valid };
                        let input: Vec<$t> = (0..source_len).map(|i| (i % 61 + 1) as $t).collect();
                        let source = DeviceBuffer::from_host(&stream, &input)?;
                        let target = DeviceBuffer::from_host(&stream, &vec![0 as $t; target_len])?;
                        let (_, target) = module
                            .$launch::<$items, _, _>(
                                &prepared, source, target, valid, shift, default, mode,
                            )
                            .sync()?;
                        let mut result = vec![0 as $t; target_len];
                        target.copy_to_host(&stream, &mut result)?;
                        stream.synchronize()?;
                        for (index, &value) in result.iter().enumerate() {
                            let within = index % ($block * 4);
                            let written = if mode == 0 || mode == 2 {
                                $block * $items
                            } else if mode == 6 {
                                valid / 2
                            } else {
                                valid
                            };
                            let expected = if within >= shift && within - shift < written {
                                if mode == 3 || (mode == 2 && within - shift >= valid) {
                                    default
                                } else {
                                    input[index]
                                }
                            } else {
                                0 as $t
                            };
                            assert_eq!(
                                value.to_ne_bytes(),
                                expected.to_ne_bytes(),
                                "type={} items={} mode={} valid={} index={}",
                                stringify!($t),
                                $items,
                                mode,
                                valid,
                                index
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }};
    }
    if let Some(trap) = trap {
        let plan = module.prepare_copy_i32_128::<4>(LaunchConfig1D::auto_shared(1, 128))?;
        let len = if trap == "ownership" { 1024 } else { 512 };
        let source = DeviceBuffer::from_host(&stream, &vec![11_i32; len])?;
        let target = DeviceBuffer::from_host(&stream, &vec![0_i32; len])?;
        let (valid, shift, mode) = match trap.as_str() {
            "unwritten" => (512, 0, 9),
            "missing-default-count" => (512, 0, 10),
            "invalid-read" => (0, 0, 11),
            "count" => (513, 0, 1),
            "bounds" => (512, 1, 1),
            "ownership" => (512, 1, 0),
            "different-count" => (127, 0, 5),
            _ => panic!("unknown trap mode"),
        };
        let _ = module
            .copy_i32_128_async_owned::<4, _, _>(&plan, source, target, valid, shift, -7, mode)
            .sync()?;
        panic!("expected trap returned successfully");
    }
    run!(
        i8,
        prepare_copy_i8_128,
        copy_i8_128_async_owned,
        128,
        1,
        i8::MIN
    );
    run!(
        i8,
        prepare_copy_i8_128,
        copy_i8_128_async_owned,
        128,
        4,
        i8::MIN
    );
    run!(
        u8,
        prepare_copy_u8_128,
        copy_u8_128_async_owned,
        128,
        1,
        u8::MAX
    );
    run!(
        u8,
        prepare_copy_u8_128,
        copy_u8_128_async_owned,
        128,
        4,
        u8::MAX
    );
    run!(
        i16,
        prepare_copy_i16_128,
        copy_i16_128_async_owned,
        128,
        1,
        i16::MIN
    );
    run!(
        i16,
        prepare_copy_i16_128,
        copy_i16_128_async_owned,
        128,
        4,
        i16::MIN
    );
    run!(
        u16,
        prepare_copy_u16_128,
        copy_u16_128_async_owned,
        128,
        1,
        u16::MAX
    );
    run!(
        u16,
        prepare_copy_u16_128,
        copy_u16_128_async_owned,
        128,
        4,
        u16::MAX
    );
    run!(
        i32,
        prepare_copy_i32_128,
        copy_i32_128_async_owned,
        128,
        1,
        i32::MIN
    );
    run!(
        i32,
        prepare_copy_i32_128,
        copy_i32_128_async_owned,
        128,
        4,
        i32::MIN
    );
    run!(
        u32,
        prepare_copy_u32_128,
        copy_u32_128_async_owned,
        128,
        1,
        u32::MAX
    );
    run!(
        u32,
        prepare_copy_u32_128,
        copy_u32_128_async_owned,
        128,
        4,
        u32::MAX
    );
    run!(
        i64,
        prepare_copy_i64_128,
        copy_i64_128_async_owned,
        128,
        1,
        i64::MIN + 17
    );
    run!(
        i64,
        prepare_copy_i64_128,
        copy_i64_128_async_owned,
        128,
        4,
        i64::MIN + 17
    );
    run!(
        u64,
        prepare_copy_u64_128,
        copy_u64_128_async_owned,
        128,
        1,
        u64::MAX - 17
    );
    run!(
        u64,
        prepare_copy_u64_128,
        copy_u64_128_async_owned,
        128,
        4,
        u64::MAX - 17
    );
    run!(
        f32,
        prepare_copy_f32_128,
        copy_f32_128_async_owned,
        128,
        1,
        -0.0_f32
    );
    run!(
        f32,
        prepare_copy_f32_128,
        copy_f32_128_async_owned,
        128,
        4,
        -0.0_f32
    );
    run!(
        f64,
        prepare_copy_f64_128,
        copy_f64_128_async_owned,
        128,
        1,
        -0.0_f64
    );
    run!(
        f64,
        prepare_copy_f64_128,
        copy_f64_128_async_owned,
        128,
        4,
        -0.0_f64
    );
    run!(
        i32,
        prepare_copy_i32_64,
        copy_i32_64_async_owned,
        64,
        1,
        -1234567
    );
    run!(
        i32,
        prepare_copy_i32_64,
        copy_i32_64_async_owned,
        64,
        4,
        -1234567
    );
    run!(
        i32,
        prepare_copy_i32_2d,
        copy_i32_2d_async_owned,
        33,
        1,
        -7,
        |groups| LaunchConfig2D::auto_shared((groups, 2), (11, 3)),
        |groups| LaunchConfig2D::auto_shared((groups, 2), (3, 11))
    );
    run!(
        i32,
        prepare_copy_i32_3d,
        copy_i32_3d_async_owned,
        128,
        1,
        -7,
        |groups| LaunchConfig3D::auto_shared((groups, 2, 2), (8, 4, 4)),
        |groups| LaunchConfig3D::auto_shared((groups, 2, 2), (16, 2, 4))
    );
    run!(
        i32,
        prepare_copy_i32_2d,
        copy_i32_2d_async_owned,
        33,
        4,
        -7,
        |groups| LaunchConfig2D::auto_shared((groups, 2), (11, 3)),
        |groups| LaunchConfig2D::auto_shared((groups, 2), (3, 11))
    );
    run!(
        i32,
        prepare_copy_i32_3d,
        copy_i32_3d_async_owned,
        128,
        4,
        -7,
        |groups| LaunchConfig3D::auto_shared((groups, 2, 2), (8, 4, 4)),
        |groups| LaunchConfig3D::auto_shared((groups, 2, 2), (16, 2, 4))
    );
    let plan = module.prepare_scalar(LaunchConfig1D::auto_shared(3, 128))?;
    let target = DeviceBuffer::from_host(&stream, &vec![0_i16; 384])?;
    let target = module
        .scalar_async_owned::<_>(&plan, target, -17_i16)
        .sync()?;
    let mut result = vec![0_i16; 384];
    target.copy_to_host(&stream, &mut result)?;
    stream.synchronize()?;
    assert!(result.iter().all(|value| *value == -17));
    cases += 1;
    println!(
        "PASS: {cases} public Direct cases, ten numeric types, scattered validity, short tails, owned async"
    );
    Ok(())
}
