// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Public Transpose checks observe values and arbitrary validity after each Store.
use cuda_coop as coop;
use cuda_device::{CollectiveTiles, DisjointSlice, kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    fn expected<T: coop::Numeric>(
        input: &[T],
        offset: usize,
        index: usize,
        items: usize,
        valid: usize,
        default: T,
        mode: u32,
    ) -> Option<T> {
        if mode == 1 || mode == 5 || mode == 6 {
            if index < valid {
                Some(input[offset + index])
            } else {
                None
            }
        } else if mode == 2 || mode == 4 {
            if index < valid {
                Some(input[offset + index])
            } else {
                Some(default)
            }
        } else if mode == 3 {
            let lane = index / items;
            let item = index % items;
            if index < valid || (lane & 1 == 0 && item & 1 == 1) || (lane & 1 == 1 && item & 1 == 0)
            {
                Some(input[offset + index])
            } else {
                None
            }
        } else {
            Some(input[offset + index])
        }
    }

    #[inline(never)]
    fn exercise<T: coop::Numeric + PartialEq, const ITEMS: usize, const CAPACITY: usize>(
        input: &[T],
        output: &mut DisjointSlice<T, CollectiveTiles<CAPACITY>>,
        valid: usize,
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
        let offset = group * threads * CAPACITY;
        let mut values = coop::ThreadData::<_, ITEMS>::new();
        if mode != 9 {
            coop::load(
                block,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .algorithm(coop::Transpose),
            );
        }
        if mode == 1 || mode == 5 || mode == 6 {
            coop::load(
                block,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .valid_items(valid)
                    .algorithm(coop::Transpose),
            );
        } else if mode == 2 || mode == 4 {
            coop::load(
                block,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .valid_items(valid)
                    .oob_default(default)
                    .algorithm(coop::Transpose),
            );
        } else if mode == 3 {
            let incoming = values;
            values = coop::ThreadData::new();
            let mut item = 0;
            while item < ITEMS {
                let index = lane * ITEMS + item;
                let initialize = index < valid
                    || (lane & 1 == 0 && item & 1 == 1)
                    || (lane & 1 == 1 && item & 1 == 0);
                if let (true, Some(value)) = (initialize, incoming.get(item)) {
                    values.write(item, value);
                }
                item += 1;
            }
        }
        if mode == 0 || mode == 2 {
            coop::store(
                block,
                &mut *output,
                &mut values,
                coop::StoreOptions::default()
                    .offset(offset)
                    .algorithm(coop::Transpose),
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
                &mut values,
                coop::StoreOptions::default()
                    .offset(offset)
                    .valid_items(written)
                    .algorithm(coop::Transpose),
            );
        }
        let mut correct = true;
        let mut item = 0;
        while item < ITEMS {
            let source = threads * item + lane;
            correct &=
                values.get(item) == expected(input, offset, source, ITEMS, valid, default, mode);
            item += 1;
        }
        // Managed scratch reuse inserts completion before this second operation.
        // A zero count still permutes all values and bits, with no global writes.
        coop::store(
            block,
            &mut *output,
            &mut values,
            coop::StoreOptions::default()
                .offset(offset)
                .valid_items(0)
                .algorithm(coop::Transpose),
        );
        item = 0;
        while item < ITEMS {
            let first = threads * item + lane;
            let source = threads * (first % ITEMS) + first / ITEMS;
            correct &=
                values.get(item) == expected(input, offset, source, ITEMS, valid, default, mode);
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
                    .oob_default(default)
                    .algorithm(coop::Transpose),
            );
            coop::store(
                block,
                &mut *output,
                &mut values,
                coop::StoreOptions::default()
                    .offset(offset)
                    .valid_items(valid)
                    .algorithm(coop::Transpose),
            );
            item = 0;
            while item < ITEMS {
                correct &= values.get(item)
                    == expected(
                        input,
                        offset,
                        threads * item + lane,
                        ITEMS,
                        valid,
                        default,
                        mode,
                    );
                item += 1;
            }
        }
        assert!(correct);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(33,1,1),dynamic_shared=inferred)]
    pub fn copy_i8_33_4<const ITEMS_PER_THREAD: usize>(
        input: &[i8],
        mut output: DisjointSlice<i8, CollectiveTiles<4>>,
        valid: usize,
        default: i8,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(33,1,1),dynamic_shared=inferred)]
    pub fn copy_u8_33_4<const ITEMS_PER_THREAD: usize>(
        input: &[u8],
        mut output: DisjointSlice<u8, CollectiveTiles<4>>,
        valid: usize,
        default: u8,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(33,1,1),dynamic_shared=inferred)]
    pub fn copy_i16_33_4<const ITEMS_PER_THREAD: usize>(
        input: &[i16],
        mut output: DisjointSlice<i16, CollectiveTiles<4>>,
        valid: usize,
        default: i16,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(33,1,1),dynamic_shared=inferred)]
    pub fn copy_u16_33_4<const ITEMS_PER_THREAD: usize>(
        input: &[u16],
        mut output: DisjointSlice<u16, CollectiveTiles<4>>,
        valid: usize,
        default: u16,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(33,1,1),dynamic_shared=inferred)]
    pub fn copy_i32_33_4<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        default: i32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(33,1,1),dynamic_shared=inferred)]
    pub fn copy_u32_33_4<const ITEMS_PER_THREAD: usize>(
        input: &[u32],
        mut output: DisjointSlice<u32, CollectiveTiles<4>>,
        valid: usize,
        default: u32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(33,1,1),dynamic_shared=inferred)]
    pub fn copy_i64_33_4<const ITEMS_PER_THREAD: usize>(
        input: &[i64],
        mut output: DisjointSlice<i64, CollectiveTiles<4>>,
        valid: usize,
        default: i64,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(33,1,1),dynamic_shared=inferred)]
    pub fn copy_u64_33_4<const ITEMS_PER_THREAD: usize>(
        input: &[u64],
        mut output: DisjointSlice<u64, CollectiveTiles<4>>,
        valid: usize,
        default: u64,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(33,1,1),dynamic_shared=inferred)]
    pub fn copy_f32_33_4<const ITEMS_PER_THREAD: usize>(
        input: &[f32],
        mut output: DisjointSlice<f32, CollectiveTiles<4>>,
        valid: usize,
        default: f32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(33,1,1),dynamic_shared=inferred)]
    pub fn copy_f64_33_4<const ITEMS_PER_THREAD: usize>(
        input: &[f64],
        mut output: DisjointSlice<f64, CollectiveTiles<4>>,
        valid: usize,
        default: f64,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(1,1,1),dynamic_shared=inferred)]
    pub fn copy_i32_1_4<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        default: i32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(128,1,1),dynamic_shared=inferred)]
    pub fn copy_i32_128_4<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        default: i32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(130,1,1),dynamic_shared=inferred)]
    pub fn copy_i32_130_4<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        default: i32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(33,1,1),dynamic_shared=inferred)]
    pub fn copy_i32_33_9<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<9>>,
        valid: usize,
        default: i32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 9>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=2,block=(11,3,1),dynamic_shared=inferred)]
    pub fn copy_i32_2d<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        default: i32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=3,block=(8,4,4),dynamic_shared=inferred)]
    pub fn copy_i32_3d<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        default: i32,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4>(input, &mut output, valid, default, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(33,1,1),dynamic_shared=inferred)]
    pub fn transpose_scalar(mut output: DisjointSlice<i16, CollectiveTiles<1>>) {
        coop::store(
            coop::this_block(),
            &mut output,
            -17_i16,
            coop::StoreOptions::default().algorithm(coop::Transpose),
        );
    }
}

pub fn run(trap: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig2D, LaunchConfig3D};
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: these typed declarations and native kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    if let Some(trap) = trap {
        let plan = module.prepare_copy_i32_33_4::<4>(LaunchConfig1D::auto_shared(1, 33))?;
        let source = DeviceBuffer::from_host(&stream, &vec![11_i32; 132])?;
        let target = DeviceBuffer::from_host(&stream, &vec![0_i32; 132])?;
        let (valid, mode) = match trap {
            "transpose-count" => (133, 1),
            "transpose-unwritten" => (132, 9),
            "transpose-different-count" => (31, 5),
            _ => panic!("unknown Transpose trap"),
        };
        let _ = module
            .copy_i32_33_4_async_owned::<4, _, _>(&plan, source, target, valid, -7, mode)
            .sync()?;
        panic!("expected Transpose trap returned successfully");
    }
    let mut cases = 0;
    macro_rules! run {
        ($t:ty,$prepare:ident,$launch:ident,$block:literal,$capacity:literal,$items:literal,$default:expr) => {
            run!(
                $t,
                $prepare,
                $launch,
                $block,
                $capacity,
                $items,
                $default,
                |groups| LaunchConfig1D::auto_shared(groups, $block),
                |groups| LaunchConfig1D::auto_shared(groups, $block + 1)
            );
        };
        ($t:ty,$prepare:ident,$launch:ident,$block:literal,$capacity:literal,$items:literal,$default:expr,
         $config:expr,$wrong_shape:expr) => {{
            let default: $t = $default;
            for groups in [1_u32, 3] {
                let plan = module.$prepare::<$items>(($config)(groups))?;
                assert!(plan.__raw_config().shared_mem_bytes > 0);
                let error = module
                    .$prepare::<$items>(($wrong_shape)(groups))
                    .err()
                    .expect("a different exact block shape must reject");
                assert!(error.to_string().contains("does not match required block"));
                let (gx, gy, gz) = plan.__raw_config().grid_dim;
                let groups = gx * gy * gz;
                for mode in [0, 1, 2, 3, 4, 6] {
                    for valid in [0, 1, $block * $items - 1, $block * $items] {
                        if mode == 0 && valid != $block * $items {
                            continue;
                        }
                        let size = groups as usize * $block * $capacity;
                        let input: Vec<$t> = (0..size).map(|i| (i % 61 + 1) as $t).collect();
                        let source = DeviceBuffer::from_host(&stream, &input)?;
                        let target = DeviceBuffer::from_host(&stream, &vec![0 as $t; size])?;
                        let (_, target) = module
                            .$launch::<$items, _, _>(&plan, source, target, valid, default, mode)
                            .sync()?;
                        let mut result = vec![0 as $t; size];
                        target.copy_to_host(&stream, &mut result)?;
                        stream.synchronize()?;
                        for (index, &value) in result.iter().enumerate() {
                            let within = index % ($block * $capacity);
                            let written = if mode == 0 || mode == 2 {
                                $block * $items
                            } else if mode == 6 {
                                valid / 2
                            } else {
                                valid
                            };
                            let expected = if within < written {
                                if mode == 2 && within >= valid {
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
                                "Transpose type={} B={} I={} mode={} valid={} index={}",
                                stringify!($t),
                                $block,
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
    run!(
        i8,
        prepare_copy_i8_33_4,
        copy_i8_33_4_async_owned,
        33,
        4,
        1,
        i8::MIN
    );
    run!(
        i8,
        prepare_copy_i8_33_4,
        copy_i8_33_4_async_owned,
        33,
        4,
        4,
        i8::MIN
    );
    run!(
        u8,
        prepare_copy_u8_33_4,
        copy_u8_33_4_async_owned,
        33,
        4,
        1,
        u8::MAX
    );
    run!(
        u8,
        prepare_copy_u8_33_4,
        copy_u8_33_4_async_owned,
        33,
        4,
        4,
        u8::MAX
    );
    run!(
        i16,
        prepare_copy_i16_33_4,
        copy_i16_33_4_async_owned,
        33,
        4,
        1,
        i16::MIN
    );
    run!(
        i16,
        prepare_copy_i16_33_4,
        copy_i16_33_4_async_owned,
        33,
        4,
        4,
        i16::MIN
    );
    run!(
        u16,
        prepare_copy_u16_33_4,
        copy_u16_33_4_async_owned,
        33,
        4,
        1,
        u16::MAX
    );
    run!(
        u16,
        prepare_copy_u16_33_4,
        copy_u16_33_4_async_owned,
        33,
        4,
        4,
        u16::MAX
    );
    run!(
        i32,
        prepare_copy_i32_33_4,
        copy_i32_33_4_async_owned,
        33,
        4,
        1,
        i32::MIN
    );
    run!(
        i32,
        prepare_copy_i32_33_4,
        copy_i32_33_4_async_owned,
        33,
        4,
        4,
        i32::MIN
    );
    run!(
        u32,
        prepare_copy_u32_33_4,
        copy_u32_33_4_async_owned,
        33,
        4,
        1,
        u32::MAX
    );
    run!(
        u32,
        prepare_copy_u32_33_4,
        copy_u32_33_4_async_owned,
        33,
        4,
        4,
        u32::MAX
    );
    run!(
        i64,
        prepare_copy_i64_33_4,
        copy_i64_33_4_async_owned,
        33,
        4,
        1,
        i64::MIN + 17
    );
    run!(
        i64,
        prepare_copy_i64_33_4,
        copy_i64_33_4_async_owned,
        33,
        4,
        4,
        i64::MIN + 17
    );
    run!(
        u64,
        prepare_copy_u64_33_4,
        copy_u64_33_4_async_owned,
        33,
        4,
        1,
        u64::MAX - 17
    );
    run!(
        u64,
        prepare_copy_u64_33_4,
        copy_u64_33_4_async_owned,
        33,
        4,
        4,
        u64::MAX - 17
    );
    run!(
        f32,
        prepare_copy_f32_33_4,
        copy_f32_33_4_async_owned,
        33,
        4,
        1,
        -0.0_f32
    );
    run!(
        f32,
        prepare_copy_f32_33_4,
        copy_f32_33_4_async_owned,
        33,
        4,
        4,
        -0.0_f32
    );
    run!(
        f64,
        prepare_copy_f64_33_4,
        copy_f64_33_4_async_owned,
        33,
        4,
        1,
        -0.0_f64
    );
    run!(
        f64,
        prepare_copy_f64_33_4,
        copy_f64_33_4_async_owned,
        33,
        4,
        4,
        -0.0_f64
    );
    run!(
        i32,
        prepare_copy_i32_1_4,
        copy_i32_1_4_async_owned,
        1,
        4,
        1,
        i32::MIN
    );
    run!(
        i32,
        prepare_copy_i32_1_4,
        copy_i32_1_4_async_owned,
        1,
        4,
        4,
        i32::MIN
    );
    run!(
        i32,
        prepare_copy_i32_128_4,
        copy_i32_128_4_async_owned,
        128,
        4,
        1,
        i32::MIN
    );
    run!(
        i32,
        prepare_copy_i32_128_4,
        copy_i32_128_4_async_owned,
        128,
        4,
        4,
        i32::MIN
    );
    run!(
        i32,
        prepare_copy_i32_130_4,
        copy_i32_130_4_async_owned,
        130,
        4,
        1,
        i32::MIN
    );
    run!(
        i32,
        prepare_copy_i32_130_4,
        copy_i32_130_4_async_owned,
        130,
        4,
        4,
        i32::MIN
    );
    run!(
        i32,
        prepare_copy_i32_33_9,
        copy_i32_33_9_async_owned,
        33,
        9,
        9,
        i32::MIN
    );
    run!(
        i32,
        prepare_copy_i32_2d,
        copy_i32_2d_async_owned,
        33,
        4,
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
        4,
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
        4,
        -7,
        |groups| LaunchConfig3D::auto_shared((groups, 2, 2), (8, 4, 4)),
        |groups| LaunchConfig3D::auto_shared((groups, 2, 2), (16, 2, 4))
    );
    let plan = module.prepare_transpose_scalar(LaunchConfig1D::auto_shared(1, 33))?;
    let target = DeviceBuffer::from_host(&stream, &[0_i16; 33])?;
    let target = module
        .transpose_scalar_async_owned::<_>(&plan, target)
        .sync()?;
    let mut result = vec![0_i16; 33];
    target.copy_to_host(&stream, &mut result)?;
    stream.synchronize()?;
    assert!(result.iter().all(|&value| value == -17));
    cases += 1;
    println!(
        "PASS: {cases} public Transpose cases, all ten types, arbitrary masks, repeated Store, partial warps and N9"
    );
    Ok(())
}
