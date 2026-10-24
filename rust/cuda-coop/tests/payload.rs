// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Public payload behavior, numeric inference, validity and element alignment.

use core::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};

use cuda_coop as coop;

#[test]
fn typed_writes_infer_every_numeric_width_at_one_and_four_items() {
    macro_rules! check {
        ($($ty:ty),* $(,)?) => {
            $(
                for value in [<$ty>::MIN, <$ty>::MAX] {
                    let mut one = coop::ThreadData::<_, 1>::new();
                    one.write(0, value);
                    assert_eq!(one.get(0), Some(value));
                    assert_eq!(size_of_val(&one[0]), size_of::<$ty>());

                    let mut four = coop::ThreadData::<_, 4>::new();
                    four.write(3, value);
                    assert_eq!(four.get(3), Some(value));
                    assert_eq!(four.get(0), None);
                    assert_eq!(four.get(1), None);
                    assert_eq!(four.get(2), None);
                }
            )*
        };
    }
    check!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64);

    let mut single = coop::ThreadData::<_, 1>::new();
    single.write(0, f32::from_bits(0x7fc0_1234));
    assert_eq!(single[0].to_bits(), 0x7fc0_1234);
    let mut double = coop::ThreadData::<_, 1>::new();
    double.write(0, f64::from_bits(0xfff8_0000_0000_1234));
    assert_eq!(double[0].to_bits(), 0xfff8_0000_0000_1234);
}

#[test]
fn arbitrary_writes_and_checked_mutation_preserve_unwritten_slots() {
    let mut items = coop::ThreadData::<_, 4>::new();
    items.write(2, 8_i32);
    items.write(0, 3_i32);
    items[2] *= 2;
    *items.get_mut(0).unwrap() += 1;
    assert_eq!(
        [items.get(0), items.get(1), items.get(2), items.get(3)],
        [Some(4), None, Some(16), None]
    );
    assert_eq!(items.len(), 4);
    assert!(!items.is_empty());
    assert!(items.get_mut(1).is_none());
    for index in [4, usize::MAX] {
        assert_eq!(items.get(index), None);
        assert!(items.get_mut(index).is_none());
    }
    items.write(2, -1);
    assert_eq!(items.get(2), Some(-1));
}

#[test]
fn sparse_writes_across_byte_boundaries_preserve_neighbors_and_copies() {
    let mut items = coop::ThreadData::<_, 33>::new();
    for index in [0, 7, 8, 16, 31, 32] {
        items.write(index, index as u16 + 1);
    }
    let copy = items;
    *items.get_mut(8).unwrap() += 10;
    items[31] += 20;
    for index in 0..33 {
        let expected = [0, 7, 8, 16, 31, 32]
            .contains(&index)
            .then_some(index as u16 + 1);
        assert_eq!(copy.get(index), expected);
        assert_eq!(
            items.get(index),
            expected.map(|value| value
                + match index {
                    8 => 10,
                    31 => 20,
                    _ => 0,
                })
        );
    }
    for index in [33, usize::MAX] {
        assert_eq!(items.get(index), None);
        assert!(items.get_mut(index).is_none());
    }
}

#[test]
fn invalid_indexing_and_writes_panic_without_initializing_other_slots() {
    let mut items = coop::ThreadData::<_, 4>::new();
    items.write(2, 9_i32);
    for index in [0, 4, usize::MAX] {
        assert!(catch_unwind(|| items[index]).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| items[index] = 17)).is_err());
    }
    for index in [4, usize::MAX] {
        assert!(catch_unwind(AssertUnwindSafe(|| items.write(index, 17))).is_err());
    }
    assert_eq!(
        [items.get(0), items.get(1), items.get(2), items.get(3)],
        [None, None, Some(9), None]
    );
}

#[test]
fn copies_moves_swaps_and_selection_keep_exact_masks() {
    let mut first = coop::ThreadData::<_, 4>::new();
    first.write(0, 1_u16);
    first.write(3, 4_u16);
    let original = first;
    let mut second = coop::ThreadData::<_, 4>::new();
    second.write(1, 2_u16);
    core::mem::swap(&mut first, &mut second);
    assert_eq!(
        (first.get(0), first.get(1), first.get(3)),
        (None, Some(2), None)
    );
    assert_eq!(
        (second.get(0), second.get(1), second.get(3)),
        (Some(1), None, Some(4))
    );
    for choose_first in [true, false] {
        let mut selected = if choose_first { first } else { second };
        assert_eq!(selected.get(1), if choose_first { Some(2) } else { None });
        selected.write(2, 99);
        assert_eq!(first.get(2), None);
        assert_eq!(second.get(2), None);
    }
    assert_eq!(
        (original.get(0), original.get(1), original.get(3)),
        (Some(1), None, Some(4))
    );
}

#[test]
fn sealed_generic_interface_preserves_inference_and_extent() {
    fn fill_last<D: coop::ThreadDataLike>(items: &mut D, value: D::Item) {
        assert_eq!(items.len(), D::EXTENT);
        assert!(!items.is_empty());
        assert!(items.get(D::EXTENT - 1).is_none());
        items.write(D::EXTENT - 1, value);
    }
    let mut items = coop::ThreadData::<_, 4, 16>::new();
    fill_last(&mut items, 255_u8);
    assert_eq!(items.get(3), Some(255_u8));
    assert_eq!(items.get(0), None);
}

#[test]
fn alignment_applies_to_the_values_array() {
    let mut items = coop::ThreadData::<_, 4, 64>::new();
    items.write(0, 13_u16);
    items.write(1, 27_u16);
    let values = &items[0] as *const u16 as usize;
    assert_eq!(values, &items as *const _ as usize);
    assert_eq!(values % 64, 0);
    assert_eq!(&items[1] as *const u16 as usize - values, size_of::<u16>());
    assert_eq!(align_of::<coop::ThreadData<u64, 1>>(), align_of::<u64>());

    // Query, rather than instantiate, the largest layouts: an over-aligned
    // object can have a large padded size even with one element.
    macro_rules! check {
        ($($align:literal),* $(,)?) => {
            $(assert_eq!(align_of::<coop::ThreadData<u8, 1, $align>>(), $align);)*
        };
    }
    check!(
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
        262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864,
        134217728, 268435456, 536870912
    );
}
