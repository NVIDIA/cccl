// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Scratch-free Warp algorithms, adjusted pointers, and shared Store preservation.
use cuda_coop as coop;
use cuda_device::{cooperative_groups as cg, kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;
    use cg::ThreadGroup as _;

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn exercise<
        T: coop::Numeric + PartialEq + From<i8> + core::ops::Add<Output = T>,
        A: coop::Algorithm + Default,
        const W: usize,
        const I: usize,
        const N: u32,
    >(
        input: &[T],
        output: &mut coop::DisjointSlice<T, coop::CollectiveTiles<8>>,
        valid: usize,
        shift: usize,
        mode: u32,
        striped: bool,
    ) where
        coop::ThreadsWithinWarp<W>: coop::LoadStoreGroup<A, ()>,
        for<'a> &'a coop::ThreadData<T, I>: coop::StoreValue<T, A>,
    {
        let group = coop::this_warp().group_by::<W>();
        let tile = cg::this_thread_block().tiled_partition::<N>();
        let lane = group.rank() as usize;
        let q = coop::this_block().rank_at::<coop::Warp>() as usize * (32 / W)
            + group.rank_at::<coop::Warp>() as usize;
        let offset = thread::blockIdx_x() as usize * 512 + q * W * (8 - I) + shift;
        let count = if mode == 0 { W * I } else { valid };
        let mut values = coop::ThreadData::<_, I>::new();
        if mode == 4 {
            let mut i = 0;
            while i < I {
                let position = if striped { lane + W * i } else { lane * I + i };
                if position < count || (lane + i) % 3 == 1 {
                    values.write(i, T::from(position as i8) + T::from(17));
                }
                i += 1;
            }
        } else {
            coop::load(
                group,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .algorithm(A::default())
                    .offset(offset),
            );
            if mode != 0 {
                let options = coop::LoadOptions::default()
                    .algorithm(A::default())
                    .offset(offset)
                    .valid_items(count);
                if mode == 2 {
                    coop::load(group, input, &mut values, options.oob_default(T::from(-7)));
                } else {
                    coop::load(group, input, &mut values, options);
                }
            }
        }
        let before = values;
        let options = coop::StoreOptions::default()
            .algorithm(A::default())
            .offset(offset);
        if mode == 0 {
            coop::store(group, &mut *output, &values, options);
        } else {
            coop::store(group, &mut *output, &values, options.valid_items(count));
        }
        let after = values;
        tile.sync();
        coop::store(
            group,
            &mut *output,
            &values,
            coop::StoreOptions::default()
                .algorithm(A::default())
                .offset(offset)
                .valid_items(0),
        );
        let twice = values;
        tile.sync();
        coop::load(
            group,
            output,
            &mut values,
            coop::LoadOptions::default()
                .algorithm(A::default())
                .offset(offset)
                .valid_items(count),
        );
        let mut matches = true;
        let mut i = 0;
        while i < I {
            let position = if striped { lane + W * i } else { lane * I + i };
            let initialized = if mode == 4 {
                position < count || (lane + i) % 3 == 1
            } else {
                mode == 0 || mode == 2 || position < count
            };
            let expected = if initialized {
                Some(if mode == 4 {
                    T::from(position as i8) + T::from(17)
                } else if mode == 2 && position >= count {
                    T::from(-7)
                } else {
                    input[offset + q * W * I + position]
                })
            } else {
                None
            };
            matches &=
                before.get(i) == expected && after.get(i) == expected && twice.get(i) == expected;
            matches &= values.get(i) == if position < count { expected } else { None };
            i += 1;
        }
        assert!(matches);
    }

    #[kernel]
    #[launch_contract(domain=1,block=(64,1,1),dynamic_shared=inferred)]
    pub fn logical_plain<
        T: coop::Numeric + PartialEq + From<i8> + core::ops::Add<Output = T>,
        A: coop::Algorithm + Default,
        const W: usize,
        const I: usize,
        const N: u32,
    >(
        input: &[T],
        mut output: coop::DisjointSlice<T, coop::CollectiveTiles<8>>,
        valid: usize,
        shift: usize,
        mode: u32,
        striped: bool,
    ) where
        coop::ThreadsWithinWarp<W>: coop::LoadStoreGroup<A, ()>,
        for<'a> &'a coop::ThreadData<T, I>: coop::StoreValue<T, A>,
    {
        exercise::<T, A, W, I, N>(input, &mut output, valid, shift, mode, striped);
    }
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    init_device_contexts(0, 1)?;
    let stream = with_cuda_context(0, |context| context.default_stream())?;
    // SAFETY: the typed module and launchers compile together.
    let module = unsafe { kernels::load_async(0)? };
    let mut cases = 0;
    macro_rules! exercise {
        ($ty:ty,$algorithm:ty,$striped:literal,$w:literal,$i:literal,$controls:expr) => {{
            let input: Vec<$ty> = (0..1536).map(|i| (i + 31) as $ty).collect();
            let plan = module.prepare_logical_plain::<$ty, $algorithm, $w, $i, $w>(
                LaunchConfig1D::auto_shared(3, 64),
            )?;
            assert_eq!(plan.__raw_config().shared_mem_bytes, 0);
            for shift in [0, 1] {
                for (mode, count) in $controls {
                    let source = DeviceBuffer::from_host(&stream, &input)?;
                    let target = DeviceBuffer::from_host(&stream, &vec![-99 as $ty; input.len()])?;
                    let (_, target) = module
                        .logical_plain_async_owned::<$ty, $algorithm, $w, $i, $w, _, _>(
                            &plan, source, target, count, shift, mode, $striped,
                        )
                        .sync()?;
                    let mut result = vec![0 as $ty; input.len()];
                    target.copy_to_host(&stream, &mut result)?;
                    stream.synchronize()?;
                    let mut expected = vec![-99 as $ty; input.len()];
                    for block in 0..3 {
                        for q in 0..64 / $w {
                            let valid = if mode == 0 { $w * $i } else { count };
                            for i in 0..valid {
                                let index = block * 512 + q * $w * 8 + shift + i;
                                expected[index] = if mode == 4 {
                                    i as $ty + 17 as $ty
                                } else {
                                    input[index]
                                };
                            }
                        }
                    }
                    assert_eq!(result, expected);
                    cases += 1;
                }
            }
        }};
    }
    let controls = |width: usize, items: usize| {
        let mut counts = vec![0, 1, width * items - 1, width * items];
        counts.sort_unstable();
        counts.dedup();
        [0, 1, 2, 4]
            .into_iter()
            .flat_map(|mode| {
                counts.iter().copied().filter_map(move |count| {
                    (mode != 0 || count == width * items).then_some((mode, count))
                })
            })
            .collect::<Vec<_>>()
    };
    exercise!(i32, coop::Striped, true, 1, 1, controls(1, 1));
    exercise!(i32, coop::Striped, true, 8, 4, controls(8, 4));
    exercise!(i32, coop::Striped, true, 32, 4, controls(32, 4));
    exercise!(i32, coop::Vectorize, false, 1, 1, controls(1, 1));
    exercise!(i32, coop::Vectorize, false, 8, 4, controls(8, 4));
    exercise!(i32, coop::Vectorize, false, 32, 4, controls(32, 4));
    exercise!(i8, coop::Striped, true, 8, 4, [(0, 32), (1, 31)]);
    exercise!(i8, coop::Vectorize, false, 8, 4, [(0, 32), (1, 31)]);
    exercise!(f64, coop::Striped, true, 8, 4, [(0, 32), (1, 31)]);
    exercise!(f64, coop::Vectorize, false, 8, 4, [(0, 32), (1, 31)]);
    println!(
        "PASS {cases} Warp Striped/Vectorize cases: shared Store preservation, offsets0/1, zero scratch"
    );
    Ok(())
}
