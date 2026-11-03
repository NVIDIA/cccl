// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! Query physical and mapped groups in an Oxide-compiled CUDA kernel.
use cuda_coop as coop;
use cuda_device::{cluster_launch, device, kernel, thread};
use cuda_host::cuda_module;
const PHYSICAL: usize = 55;
const MAPPED: usize = 20;

// Output addresses use raw CUDA coordinates, independently of group queries.
#[device]
fn output_index() -> usize {
    let block = (u64::from(thread::blockIdx_z()) * u64::from(thread::gridDim_y())
        + u64::from(thread::blockIdx_y()))
        * u64::from(thread::gridDim_x())
        + u64::from(thread::blockIdx_x());
    let lane = (thread::threadIdx_z() * thread::blockDim_y() + thread::threadIdx_y())
        * thread::blockDim_x()
        + thread::threadIdx_x();
    let threads = thread::blockDim_x() * thread::blockDim_y() * thread::blockDim_z();
    (block * u64::from(threads) + u64::from(lane)) as usize
}

#[device]
unsafe fn payload_values<const ITEMS: usize>(output: *mut i32) {
    let lane = output_index();
    let selected = lane % ITEMS;
    let seed = lane as i32 * 3 - 11;
    let mut values = coop::ThreadData::<_, ITEMS>::new();
    for i in 0..ITEMS {
        assert!(values.get(i).is_none());
        assert!(values.get_mut(i).is_none());
    }
    values.write(selected, seed);
    let mut copy = values;
    *copy.get_mut(selected).unwrap() += 7;
    copy[selected] += 5;
    assert_eq!(values[selected], seed);
    assert!(copy.get(ITEMS).is_none());
    for i in 0..ITEMS {
        values.write(i, seed + i as i32);
        unsafe {
            output.add(lane * ITEMS * 2 + i).write(values[i]);
            output
                .add(lane * ITEMS * 2 + ITEMS + i)
                .write(copy.get(i).unwrap_or(-999));
        }
    }
}

// Check each local query's public return width while converting test output.
macro_rules! local {
    ($query:expr) => {{
        let value: u32 = $query;
        u64::from(value)
    }};
}

#[device]
unsafe fn physical(output: *mut u64) {
    let hierarchy = coop::Hierarchy::current();
    let thread = hierarchy.thread();
    let warp = hierarchy.warp();
    let block = hierarchy.block();
    let cluster = hierarchy.cluster();
    let grid = hierarchy.grid();
    let values = [
        local!(thread.rank_at::<coop::Thread>()),
        local!(thread.count_at::<coop::Thread>()),
        local!(thread.rank_at::<coop::Warp>()),
        local!(thread.count_at::<coop::Warp>()),
        local!(thread.rank_at::<coop::Block>()),
        local!(thread.count_at::<coop::Block>()),
        local!(thread.rank_at::<coop::Cluster>()),
        local!(thread.count_at::<coop::Cluster>()),
        thread.rank_at::<coop::Grid>(),
        thread.count_at::<coop::Grid>(),
        local!(warp.rank_at::<coop::Thread>()),
        local!(warp.count_at::<coop::Thread>()),
        local!(warp.rank_at::<coop::Warp>()),
        local!(warp.count_at::<coop::Warp>()),
        local!(warp.rank_at::<coop::Block>()),
        local!(warp.count_at::<coop::Block>()),
        local!(warp.rank_at::<coop::Cluster>()),
        local!(warp.count_at::<coop::Cluster>()),
        warp.rank_at::<coop::Grid>(),
        warp.count_at::<coop::Grid>(),
        local!(block.rank_at::<coop::Thread>()),
        local!(block.count_at::<coop::Thread>()),
        local!(block.rank_at::<coop::Warp>()),
        local!(block.count_at::<coop::Warp>()),
        local!(block.rank_at::<coop::Block>()),
        local!(block.count_at::<coop::Block>()),
        local!(block.rank_at::<coop::Cluster>()),
        local!(block.count_at::<coop::Cluster>()),
        block.rank_at::<coop::Grid>(),
        block.count_at::<coop::Grid>(),
        local!(cluster.rank_at::<coop::Thread>()),
        local!(cluster.count_at::<coop::Thread>()),
        local!(cluster.rank_at::<coop::Warp>()),
        local!(cluster.count_at::<coop::Warp>()),
        local!(cluster.rank_at::<coop::Block>()),
        local!(cluster.count_at::<coop::Block>()),
        local!(cluster.rank_at::<coop::Cluster>()),
        local!(cluster.count_at::<coop::Cluster>()),
        cluster.rank_at::<coop::Grid>(),
        cluster.count_at::<coop::Grid>(),
        grid.rank_at::<coop::Thread>(),
        grid.count_at::<coop::Thread>(),
        grid.rank_at::<coop::Warp>(),
        grid.count_at::<coop::Warp>(),
        grid.rank_at::<coop::Block>(),
        grid.count_at::<coop::Block>(),
        grid.rank_at::<coop::Cluster>(),
        grid.count_at::<coop::Cluster>(),
        grid.rank_at::<coop::Grid>(),
        grid.count_at::<coop::Grid>(),
        thread.is_member() as u64,
        warp.is_member() as u64,
        block.is_member() as u64,
        cluster.is_member() as u64,
        grid.is_member() as u64,
    ];
    let base = output_index() * PHYSICAL;
    for (i, value) in values.into_iter().enumerate() {
        unsafe {
            output.add(base + i).write(value);
        }
    }
}

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub unsafe fn queries(output: *mut u64) {
        unsafe {
            physical(output);
        }
    }
    #[kernel]
    #[cluster_launch(2, 2, 1)]
    pub unsafe fn clustered(output: *mut u64) {
        unsafe {
            physical(output);
        }
    }
    #[kernel]
    pub unsafe fn mapped(output: *mut u64) {
        let lanes = coop::this_warp().group_by_non_exhaustive::<6>();
        let warps = coop::this_block().group_by_non_exhaustive::<2>();
        let exact = coop::this_warp().group_by::<8>();
        let single = coop::this_block().group_by::<1>();
        let lane_member: u8 = lanes.is_member();
        let warp_member: u8 = warps.is_member();
        let values = [
            lane_member as u64,
            lanes.count() as u64,
            local!(lanes.count_at::<coop::Warp>()),
            if lane_member == 1 {
                lanes.rank() as u64
            } else {
                u64::MAX
            },
            if lane_member == 1 {
                local!(lanes.rank_at::<coop::Warp>())
            } else {
                u64::MAX
            },
            warp_member as u64,
            warps.count() as u64,
            local!(warps.count_at::<coop::Warp>()),
            local!(warps.count_at::<coop::Block>()),
            if warp_member == 1 {
                warps.rank() as u64
            } else {
                u64::MAX
            },
            if warp_member == 1 {
                local!(warps.rank_at::<coop::Warp>())
            } else {
                u64::MAX
            },
            if warp_member == 1 {
                local!(warps.rank_at::<coop::Block>())
            } else {
                u64::MAX
            },
            exact.rank_as::<i8>() as u64,
            exact.count_as::<u8>() as u64,
            exact.rank_at_as::<i16, coop::Warp>() as u64,
            exact.count_at_as::<u16, coop::Warp>() as u64,
            single.rank_as::<i32>() as u64,
            single.count_as::<u32>() as u64,
            single.rank_at_as::<i64, coop::Block>() as u64,
            single.count_at_as::<u64, coop::Block>(),
        ];
        let base = output_index() * MAPPED;
        for (i, value) in values.into_iter().enumerate() {
            unsafe {
                output.add(base + i).write(value);
            }
        }
    }
    #[kernel]
    pub unsafe fn payload(output: *mut i32, items: u32) {
        unsafe {
            if items == 1 {
                payload_values::<1>(output);
            } else {
                payload_values::<4>(output);
            }
        }
    }
    // Each negative case runs in a separate process because a device trap
    // makes the CUDA context unusable for subsequent launches.
    #[kernel]
    pub unsafe fn invalid(output: *mut u32, mode: u32) {
        let value = match mode {
            0 => coop::this_warp().group_by::<8>().count(),
            1 => coop::this_block().group_by::<2>().count(),
            2 => coop::this_block().group_by_non_exhaustive::<4>().count(),
            3 => coop::this_warp().group_by_non_exhaustive::<6>().rank(),
            _ => {
                let mut data = coop::ThreadData::<_, 4>::new();
                data.write(0, 1_u32);
                data[1]
            }
        };
        unsafe {
            output.add(output_index()).write(value);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    if let Some(mode) = std::env::args().nth(1) {
        let mode: u32 = mode.parse()?;
        assert!(mode <= 4);
        let threads = if mode == 0 { 48 } else { 96 };
        let out = DeviceBuffer::<u32>::zeroed(&stream, threads)?;
        unsafe {
            module.invalid(
                &stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                out.cu_deviceptr() as *mut u32,
                mode,
            )?;
        }
        let err = out
            .to_host_vec(&stream)
            .expect_err("invalid group query or payload access must trap");
        assert!(err.to_string().contains("719"), "unexpected error: {err}");
        println!("PASS rejected invalid query/payload case {mode}: {err}");
        return Ok(());
    }
    let grid = (4u32, 2u32, 2u32);
    for (block, clustered) in [((8, 3, 4), false), ((8, 3, 2), false), ((8, 3, 4), true)] {
        let nt = block.0 * block.1 * block.2;
        let nb = grid.0 * grid.1 * grid.2;
        let out = DeviceBuffer::<u64>::zeroed(&stream, (nt * nb) as usize * PHYSICAL)?;
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };
        unsafe {
            if clustered {
                module.clustered(&stream, cfg, out.cu_deviceptr() as *mut u64)?;
            } else {
                module.queries(&stream, cfg, out.cu_deviceptr() as *mut u64)?;
            }
        }
        let values = out.to_host_vec(&stream)?;
        let c = if clustered { (2, 2, 1) } else { (1, 1, 1) };
        let cb = c.0 * c.1 * c.2;
        for bz in 0..grid.2 {
            for by in 0..grid.1 {
                for bx in 0..grid.0 {
                    let b = (bz * grid.1 + by) * grid.0 + bx;
                    let br = ((bz % c.2) * c.1 + by % c.1) * c.0 + bx % c.0;
                    let cr = ((bz / c.2) * (grid.1 / c.1) + by / c.1) * (grid.0 / c.0) + bx / c.0;
                    for t in 0..nt {
                        let nw = nt.div_ceil(32);
                        let mut expected = [1u64; PHYSICAL];
                        let pairs = [
                            (0, 1, t % 32, 32),
                            (0, 2, t, nt),
                            (1, 2, t / 32, nw),
                            (0, 3, br * nt + t, cb * nt),
                            (1, 3, br * nw + t / 32, cb * nw),
                            (2, 3, br, cb),
                            (0, 4, b * nt + t, nb * nt),
                            (1, 4, b * nw + t / 32, nb * nw),
                            (2, 4, b, nb),
                            (3, 4, cr, nb / cb),
                        ];
                        for i in 0..5 {
                            expected[(i * 5 + i) * 2] = 0;
                        }
                        for (inner, outer, rank, count) in pairs {
                            for (s, l) in [(inner, outer), (outer, inner)] {
                                expected[(s * 5 + l) * 2] = u64::from(rank);
                                expected[(s * 5 + l) * 2 + 1] = u64::from(count);
                            }
                        }
                        let start = (b * nt + t) as usize * PHYSICAL;
                        assert_eq!(
                            &values[start..start + PHYSICAL],
                            expected,
                            "block={b}, thread={t}"
                        );
                    }
                }
            }
        }
        println!(
            "PASS physical grid={grid:?} block={block:?} cluster={c:?}: {} values",
            values.len()
        );
    }
    for block in [(8, 3, 4), (8, 5, 4)] {
        let nt = block.0 * block.1 * block.2;
        let nb = grid.0 * grid.1 * grid.2;
        let out = DeviceBuffer::<u64>::zeroed(&stream, (nt * nb) as usize * MAPPED)?;
        unsafe {
            module.mapped(
                &stream,
                LaunchConfig {
                    grid_dim: grid,
                    block_dim: block,
                    shared_mem_bytes: 0,
                },
                out.cu_deviceptr() as *mut u64,
            )?;
        }
        let values = out.to_host_vec(&stream)?;
        for b in 0..nb {
            for t in 0..nt {
                let lane = t % 32;
                let warp = t / 32;
                let nw = nt / 32;
                let lm = lane < 30;
                let wm = warp < nw / 2 * 2;
                let expected = [
                    lm as u64,
                    6,
                    5,
                    if lm { (lane % 6) as u64 } else { u64::MAX },
                    if lm { (lane / 6) as u64 } else { u64::MAX },
                    wm as u64,
                    64,
                    2,
                    (nw / 2) as u64,
                    if wm {
                        (warp % 2 * 32 + lane) as u64
                    } else {
                        u64::MAX
                    },
                    if wm { (warp % 2) as u64 } else { u64::MAX },
                    if wm { (warp / 2) as u64 } else { u64::MAX },
                    (lane % 8) as u64,
                    8,
                    (lane / 8) as u64,
                    4,
                    lane as u64,
                    32,
                    warp as u64,
                    nw as u64,
                ];
                let start = (b * nt + t) as usize * MAPPED;
                assert_eq!(
                    &values[start..start + MAPPED],
                    expected,
                    "mapped block={b}, thread={t}"
                );
            }
        }
        println!("PASS mapped block={block:?}: {} values", values.len());
    }
    for items in [1, 4] {
        let block = (8, 3, 4);
        let threads = block.0 * block.1 * block.2 * grid.0 * grid.1 * grid.2;
        let out = DeviceBuffer::<i32>::zeroed(&stream, threads as usize * items * 2)?;
        unsafe {
            module.payload(
                &stream,
                LaunchConfig {
                    grid_dim: grid,
                    block_dim: block,
                    shared_mem_bytes: 0,
                },
                out.cu_deviceptr() as *mut i32,
                items as u32,
            )?;
        }
        let actual = out.to_host_vec(&stream)?;
        for lane in 0..threads as usize {
            let seed = lane as i32 * 3 - 11;
            for i in 0..items {
                assert_eq!(actual[lane * items * 2 + i], seed + i as i32);
                assert_eq!(
                    actual[lane * items * 2 + items + i],
                    if i == lane % items { seed + 12 } else { -999 }
                );
            }
        }
        println!("PASS payload items={items}: {} values", actual.len());
    }
    Ok(())
}
