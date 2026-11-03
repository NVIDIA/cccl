// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private native-boundary bridge. These addresses describe opaque
// MaybeUninit cells and a separate packed validity bitmap, never a T slice.
use super::{Alignment, Numeric, SupportedAlignment, ThreadData};

impl<T: Numeric, const ITEMS: usize, const ALIGN: usize> ThreadData<T, ITEMS, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    #[inline(always)]
    pub(crate) fn block_io_read_parts(&self) -> (*const T, *const u8) {
        let () = Self::POSITIVE_EXTENT;
        (self.values.as_ptr().cast::<T>(), self.initialized.as_ptr())
    }

    #[inline(always)]
    pub(crate) fn block_io_parts(&mut self) -> (*mut T, *mut u8) {
        let () = Self::POSITIVE_EXTENT;
        (
            self.values.as_mut_ptr().cast::<T>(),
            self.initialized.as_mut_ptr(),
        )
    }
}
