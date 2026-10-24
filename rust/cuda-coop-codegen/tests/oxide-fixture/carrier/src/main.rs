// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use coop_marker_fixture_api::carrier;
use cuda_device::{CollectiveTiles, DisjointSlice, kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn cycle<const ITEMS: usize>(
        input: &[i32],
        output: &mut DisjointSlice<i32, CollectiveTiles<4>>,
        offset: usize,
        valid: usize,
    ) {
        let mut values = [0; ITEMS];
        carrier::load(input, offset, valid, &mut values);
        carrier::store(output, offset, valid, &mut values);
        thread::sync_threads();
        carrier::reload(output, offset, valid, &mut values);
        thread::sync_threads();
        carrier::store(output, offset, valid, &mut values);
        thread::sync_threads();
    }

    #[inline(never)]
    fn repeat<const ITEMS: usize>(
        input: &[i32],
        output: &mut DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        rounds: u32,
    ) {
        let rank = thread::blockIdx_x() as usize
            + thread::gridDim_x() as usize
                * (thread::blockIdx_y() as usize
                    + thread::gridDim_y() as usize * thread::blockIdx_z() as usize);
        let offset = rank * 128 * 4 + shift;
        let mut remaining = rounds;
        while remaining > 0 {
            cycle::<ITEMS>(input, output, offset, valid);
            remaining -= 1;
        }
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_one(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        rounds: u32,
    ) {
        repeat::<1>(input, &mut output, valid, shift, rounds);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_four(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        shift: usize,
        rounds: u32,
    ) {
        repeat::<4>(input, &mut output, valid, shift, rounds);
    }
    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn changed_items(input: &[i32], mut output: DisjointSlice<i32, CollectiveTiles<4>>) {
        repeat::<1>(input, &mut output, 128, 0, 1);
        let offset = thread::blockIdx_x() as usize * 512;
        let mut values = [0; 4];
        carrier::load(input, offset, 512, &mut values);
        carrier::store(&mut output, offset, 512, &mut values);
        // No final barrier: returning the owned buffers waits for this kernel.
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn repartitioned(input: &[i32], mut output: DisjointSlice<i32, CollectiveTiles<4>>) {
        // ITEMS=1 does not change the capacity-4 ownership grain. Block 1's
        // offset is in bounds but belongs to block 0.
        let offset = thread::blockIdx_x() as usize * 128;
        cycle::<1>(input, &mut output, offset, 128);
    }

    #[kernel]
    #[launch_contract(domain = 2, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn copy_four_2d(input: &[i32], mut output: DisjointSlice<i32, CollectiveTiles<4>>) {
        repeat::<4>(input, &mut output, 495, 17, 2);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig2D};

    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: this module contains the declarations above.
    let module = unsafe { kernels::load_async(0)? };
    let config = LaunchConfig1D::auto_shared(3, 128);
    let one = module.prepare_copy_one(config)?;
    let four = module.prepare_copy_four(config)?;
    assert_eq!(one.__raw_config().shared_mem_bytes, 0);
    let input: Vec<_> = (0..1536).map(|i| i * 3 + 17).collect();
    if let Some(mode) = std::env::args().nth(1) {
        // All shifted ranges fit the allocation in the overlap case. Only
        // canonical group containment forbids the adjacent-block overlap.
        let source = DeviceBuffer::from_host(&stream, &vec![1_i32; 2048])?;
        let target = DeviceBuffer::from_host(&stream, &vec![0_i32; 2048])?;
        if mode == "repartition" {
            let repartitioned = module.prepare_repartitioned(config)?;
            let _ = module
                .repartitioned_async_owned(&repartitioned, source, target)
                .sync()?;
        } else {
            let shift = match mode.as_str() {
                "overlap" => 1,
                "bounds" => 2048,
                _ => return Err(format!("unknown trap mode: {mode}").into()),
            };
            let _ = module
                .copy_four_async_owned(&four, source, target, 512, shift, 1)
                .sync()?;
        }
        panic!("invalid collective range did not trap");
    }
    for items in [1, 4] {
        for shift in [0, 17] {
            let limit = (items * 128).min(512 - shift);
            for valid in [0, 1, limit - 1, limit] {
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let target = DeviceBuffer::from_host(&stream, &vec![-99; input.len()])?;
                let (_source, target) = if items == 1 {
                    module
                        .copy_one_async_owned(&one, source, target, valid, shift, 2)
                        .sync()?
                } else {
                    module
                        .copy_four_async_owned(&four, source, target, valid, shift, 2)
                        .sync()?
                };
                let mut result = vec![0; input.len()];
                target.copy_to_host(&stream, &mut result)?;
                stream.synchronize()?;
                for (index, value) in result.into_iter().enumerate() {
                    let expected = if index % 512 >= shift && index % 512 - shift < valid {
                        input[index]
                    } else {
                        -99
                    };
                    assert_eq!(
                        value, expected,
                        "items={items}, valid={valid}, index={index}"
                    );
                }
            }
        }
    }
    let changed = module.prepare_changed_items(config)?;
    let source = DeviceBuffer::from_host(&stream, &input)?;
    let target = DeviceBuffer::from_host(&stream, &vec![-99; input.len()])?;
    let (_source, target) = module
        .changed_items_async_owned(&changed, source, target)
        .sync()?;
    let mut result = vec![0; input.len()];
    target.copy_to_host(&stream, &mut result)?;
    stream.synchronize()?;
    assert_eq!(result, input);

    let two_dimensional =
        module.prepare_copy_four_2d(LaunchConfig2D::auto_shared((2, 2), (128, 1)))?;
    let input_2d: Vec<_> = (0..2048).map(|i| 11 * i - 23).collect();
    let source = DeviceBuffer::from_host(&stream, &input_2d)?;
    let target = DeviceBuffer::from_host(&stream, &vec![-99; input_2d.len()])?;
    let (_source, target) = module
        .copy_four_2d_async_owned(&two_dimensional, source, target)
        .sync()?;
    let mut result = vec![0; input_2d.len()];
    target.copy_to_host(&stream, &mut result)?;
    stream.synchronize()?;
    for (index, value) in result.into_iter().enumerate() {
        let expected = if index % 512 >= 17 {
            input_2d[index]
        } else {
            -99
        };
        assert_eq!(value, expected, "2D grid index={index}");
    }
    println!("PASS: 28,160 carrier values, bounds, item counts, 2D grid, reuse, owned async");
    Ok(())
}
