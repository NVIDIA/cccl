// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Construct and inspect a per-thread payload on the host.

use cuda_coop as coop;

fn payload<const ITEMS_PER_THREAD: usize>(value: i16) -> coop::ThreadData<i16, ITEMS_PER_THREAD> {
    let mut items = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
    items.write(0, value);
    items
}

fn main() {
    let mut items = payload::<4>(7);
    assert_eq!(items.get(0), Some(7_i16));
    assert_eq!(items.get(1), None);
    items[0] += 2;
    items.write(3, 15);
    assert_eq!(items.get(0), Some(9));
    assert_eq!(items.get(3), Some(15));
}
