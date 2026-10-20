// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private native-boundary bridge, compiled only by the coordinated fixture
// until the complete collective API is qualified. These addresses describe
// opaque MaybeUninit cells and separate exact-one flags, never a T slice.
use super::{Alignment, Numeric, SupportedAlignment, ThreadData};

impl<T: Numeric, const ITEMS: usize, const ALIGN: usize> ThreadData<T, ITEMS, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    #[inline(always)]
    pub(crate) fn block_io_parts(&mut self) -> (*mut T, *mut u8) {
        let () = Self::POSITIVE_EXTENT;
        (
            self.values.as_mut_ptr().cast::<T>(),
            self.initialized.as_mut_ptr(),
        )
    }
}
