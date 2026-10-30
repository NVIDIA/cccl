// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use coop_marker_fixture_helper::direct_load_filled;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{kernel, launch_contract};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(block = (128, 1, 1), domain = 3, dynamic_shared = inferred)]
    pub unsafe fn probe(input: *const i32, counter: *const DeviceAtomicU32, flag: bool) {
        // A panic-only host placeholder still returns a varying device value.
        // Merging it with a constant branch must not prove uniformity.
        let count = if flag {
            unsafe { (*counter).fetch_add(1, AtomicOrdering::Relaxed) as i32 }
        } else {
            0
        };
        let mut value = 0_i32;
        unsafe {
            cuda_device::library::checked_block_call_v1::<i32, _, _, _, 1>(
                direct_load_filled,
                (input, count, -7),
                (&mut value as *mut i32,),
            );
        }
    }
}

fn main() {}
