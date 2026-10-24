// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 3, 2), dynamic_shared = inferred)]
    pub fn flag_tile<const ITEMS_PER_THREAD: usize>(
        input: &[f64],
        mut heads: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut tails: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        predecessor: f64,
        successor: f64,
    ) {
        let block = coop::this_block();
        let offset =
            block.rank_at::<coop::Grid>() as usize * block.count() as usize * ITEMS_PER_THREAD;
        let (head_flags, tail_flags, values) = coop_discontinuity_helper::flag_tile::<
            _,
            ITEMS_PER_THREAD,
        >(
            block, input, offset, predecessor, successor
        );
        coop::store(
            block,
            &mut heads,
            &head_flags,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut tails,
            &tail_flags,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut preserved,
            &values,
            coop::StoreOptions::default().offset(offset),
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig3D};

    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: the typed declarations and embedded kernels compile together.
    let module = unsafe { kernels::load_async(0)? };
    macro_rules! run {
        ($items:literal) => {{
            let tile = 24 * $items;
            let input: Vec<f64> = (0..3 * tile)
                .map(|i| ((i % tile / 3 + 2 * (i / tile)) % 7) as f64 + 0.25)
                .collect();
            let predecessor = 0.25_f64;
            let successor = 3.25_f64;
            let mut expected_heads = Vec::with_capacity(input.len());
            let mut expected_tails = Vec::with_capacity(input.len());
            for values in input.chunks_exact(tile) {
                for (i, &value) in values.iter().enumerate() {
                    let previous = if i == 0 { predecessor } else { values[i - 1] };
                    let next = if i + 1 == tile {
                        successor
                    } else {
                        values[i + 1]
                    };
                    expected_heads.push(i32::from(previous != value));
                    expected_tails.push(i32::from(value != next));
                }
            }
            let source = DeviceBuffer::from_host(&stream, &input)?;
            let head_target = DeviceBuffer::from_host(&stream, &vec![-1_i32; input.len()])?;
            let tail_target = DeviceBuffer::from_host(&stream, &vec![-1_i32; input.len()])?;
            let preserved_target = DeviceBuffer::from_host(&stream, &vec![-1_f64; input.len()])?;
            let plan = module
                .prepare_flag_tile::<$items>(LaunchConfig3D::auto_shared((3, 1, 1), (4, 3, 2)))?;
            let (source, head_target, tail_target, preserved_target) = module
                .flag_tile_async_owned::<$items, _, _, _, _>(
                    &plan,
                    source,
                    head_target,
                    tail_target,
                    preserved_target,
                    predecessor,
                    successor,
                )
                .sync()?;
            let mut heads = vec![0_i32; input.len()];
            let mut tails = vec![0_i32; input.len()];
            let mut preserved = vec![0_f64; input.len()];
            let mut source_after = vec![0_f64; input.len()];
            head_target.copy_to_host(&stream, &mut heads)?;
            tail_target.copy_to_host(&stream, &mut tails)?;
            preserved_target.copy_to_host(&stream, &mut preserved)?;
            source.copy_to_host(&stream, &mut source_after)?;
            stream.synchronize()?;
            assert_eq!(heads, expected_heads);
            assert_eq!(tails, expected_tails);
            for i in 0..input.len() {
                assert_eq!(preserved[i].to_bits(), input[i].to_bits());
                assert_eq!(source_after[i].to_bits(), input[i].to_bits());
            }
            println!(
                "items/thread={}: {} head flags, {} tail flags and {} preserved inputs checked",
                $items,
                heads.len(),
                tails.len(),
                preserved.len()
            );
        }};
    }
    run!(1);
    run!(4);
    Ok(())
}
