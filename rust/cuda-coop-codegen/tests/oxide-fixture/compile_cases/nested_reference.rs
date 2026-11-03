// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
use coop_marker_fixture_helper::{direct_load_filled, direct_load_full};
use cuda_device::{kernel, launch_contract};
use cuda_host::cuda_module;
#[inline(never)]
fn twice(
    value: &&i32,
    _a: i32,
    _b: i32,
    _c: i32,
    _d: i32,
    _e: i32,
    _f: i32,
    _g: i32,
    _h: i32,
) -> i32 {
    **value
}
#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    #[launch_contract(block = (128,1,1), domain = 3, dynamic_shared = inferred)]
    pub unsafe fn probe(input: *const i32) {
        let value = cuda_device::thread::threadIdx_x() as i32;
        let reference = &value;
        let count = twice(&reference, 0, 0, 0, 0, 0, 0, 0, 0);
        let mut payload = 0_i32;
        unsafe {
            cuda_device::library::checked_block_call_v1::<i32, _, _, _, 1>(
                direct_load_filled,
                (input, count, -7_i32),
                (&mut payload as *mut i32,),
            );
        }
    }
}
fn main() {}
