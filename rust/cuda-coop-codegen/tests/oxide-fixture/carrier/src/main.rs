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
        let offset = thread::blockIdx_x() as usize * 128 * 4 + shift;
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};

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
        let shift = if mode == "overlap" { 1 } else { 2048 };
        let _ = module
            .copy_four_async_owned(&four, source, target, 512, shift, 1)
            .sync()?;
        panic!("invalid collective range did not trap");
    }
    for items in [1, 4] {
        for shift in [0, 17] {
            for valid in [0, 1, items * 128 - 1, items * 128] {
                if valid > 512 - shift {
                    continue;
                }
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
    println!("PASS: carrier bounds, item counts, repeated Store/barrier/Load, owned async");
    Ok(())
}
