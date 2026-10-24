// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Private unsafe fixture wrappers, with no public collective API contract.
#![no_std]

#[cfg(feature = "carrier")]
pub mod carrier;

// Test-source facade: compile the exact payload/group implementation without
// introducing a public operation or a second representation of ThreadData.
#[cfg(feature = "validity")]
mod alignment {
    include!(concat!(
        env!("CCCL_SOURCE"),
        "/rust/cuda-coop/src/alignment.rs"
    ));
}
#[cfg(feature = "validity")]
pub use alignment::{Alignment, SupportedAlignment};
#[cfg(feature = "validity")]
pub mod thread_data {
    include!(concat!(
        env!("CCCL_SOURCE"),
        "/rust/cuda-coop/src/thread_data.rs"
    ));

    // Fault injection and flag observation belong only to this private fixture.
    pub fn flag<T: Numeric, const N: usize>(values: &ThreadData<T, N>, index: usize) -> u8 {
        (values.initialized[index / 8] >> (index % 8)) & 1
    }
    pub fn clear_slot<T: Numeric, const N: usize>(values: &mut ThreadData<T, N>, index: usize) {
        values.initialized[index / 8] &= !(1 << (index % 8));
    }
}
#[cfg(feature = "validity")]
pub use thread_data::{Numeric, ThreadData};
#[cfg(feature = "validity")]
mod group {
    include!(concat!(env!("CCCL_SOURCE"), "/rust/cuda-coop/src/group.rs"));
}
#[cfg(feature = "validity")]
pub use group::*;
#[cfg(feature = "validity")]
pub mod validity;

macro_rules! registration {
    ($name:ident, $variable:literal) => {
        fn $name() -> &'static str {
            cuda_device::library::registration!($variable)
        }
    };
}
registration!(direct_load_full, "COOP_DIRECT_LOAD_FULL");
registration!(direct_load_filled, "COOP_DIRECT_LOAD_FILLED");
registration!(direct_store_full, "COOP_DIRECT_STORE_FULL");
registration!(direct_store_partial, "COOP_DIRECT_STORE_PARTIAL");
registration!(transpose_load_full, "COOP_TRANSPOSE_LOAD_FULL");
registration!(transpose_load_filled, "COOP_TRANSPOSE_LOAD_FILLED");
registration!(transpose_store_full, "COOP_TRANSPOSE_STORE_FULL");
registration!(transpose_store_partial, "COOP_TRANSPOSE_STORE_PARTIAL");
#[cfg(feature = "validity")]
registration!(validity_load_full, "COOP_VALIDITY_LOAD_FULL");
#[cfg(feature = "validity")]
registration!(validity_load_masked, "COOP_VALIDITY_LOAD_MASKED");
#[cfg(feature = "validity")]
registration!(validity_load_filled, "COOP_VALIDITY_LOAD_FILLED");
#[cfg(feature = "validity")]
registration!(validity_store_full, "COOP_VALIDITY_STORE_FULL");
#[cfg(feature = "validity")]
registration!(validity_store_partial, "COOP_VALIDITY_STORE_PARTIAL");

/// # Safety
/// All threads participate with one block-wide input origin, valid count and
/// fill selection. The caller satisfies the exact CUB ABI and collective requirements.
#[inline(never)]
pub unsafe fn load<const ITEMS_PER_THREAD: usize, const TRANSPOSE: bool>(
    input: *const i32,
    values: *mut i32,
    valid: i32,
    filled: bool,
) {
    unsafe {
        if filled {
            if TRANSPOSE {
                cuda_device::library::call_v1::<i32, _, _, 128, ITEMS_PER_THREAD>(
                    transpose_load_filled,
                    (input, values, valid, -7_i32),
                );
            } else {
                cuda_device::library::call_v1::<i32, _, _, 128, ITEMS_PER_THREAD>(
                    direct_load_filled,
                    (input, values, valid, -7_i32),
                );
            }
        } else if TRANSPOSE {
            cuda_device::library::call_v1::<i32, _, _, 128, ITEMS_PER_THREAD>(
                transpose_load_full,
                (input, values),
            );
        } else {
            cuda_device::library::call_v1::<i32, _, _, 128, ITEMS_PER_THREAD>(
                direct_load_full,
                (input, values),
            );
        }
    }
}

/// # Safety
/// The payload is initialized. All threads participate with one block-wide
/// output origin, valid count and partial selection, satisfying the exact CUB ABI.
#[inline(never)]
pub unsafe fn store<const ITEMS_PER_THREAD: usize, const TRANSPOSE: bool>(
    output: *mut i32,
    values: *mut i32,
    valid: i32,
    partial: bool,
) {
    unsafe {
        if partial {
            if TRANSPOSE {
                cuda_device::library::call_v1::<i32, _, _, 128, ITEMS_PER_THREAD>(
                    transpose_store_partial,
                    (output, values, valid),
                );
            } else {
                cuda_device::library::call_v1::<i32, _, _, 128, ITEMS_PER_THREAD>(
                    direct_store_partial,
                    (output, values, valid),
                );
            }
        } else if TRANSPOSE {
            cuda_device::library::call_v1::<i32, _, _, 128, ITEMS_PER_THREAD>(
                transpose_store_full,
                (output, values),
            );
        } else {
            cuda_device::library::call_v1::<i32, _, _, 128, ITEMS_PER_THREAD>(
                direct_store_full,
                (output, values),
            );
        }
    }
}

#[cfg(feature = "checked")]
mod checked {
    use super::*;
    /// # Safety
    /// All threads participate with one block-wide input origin, valid count and
    /// fill selection. The caller satisfies the exact CUB ABI and collective requirements.
    #[inline(never)]
    pub unsafe fn checked_load<const ITEMS_PER_THREAD: usize, const TRANSPOSE: bool>(
        input: *const i32,
        values: *mut i32,
        valid: i32,
        filled: bool,
    ) {
        unsafe {
            if filled {
                if TRANSPOSE {
                    cuda_device::library::checked_block_call_v1::<i32, _, _, _, ITEMS_PER_THREAD>(
                        transpose_load_filled,
                        (input, valid, -7_i32),
                        (values,),
                    );
                } else {
                    cuda_device::library::checked_block_call_v1::<i32, _, _, _, ITEMS_PER_THREAD>(
                        direct_load_filled,
                        (input, valid, -7_i32),
                        (values,),
                    );
                }
            } else if TRANSPOSE {
                cuda_device::library::checked_block_call_v1::<i32, _, _, _, ITEMS_PER_THREAD>(
                    transpose_load_full,
                    (input,),
                    (values,),
                );
            } else {
                cuda_device::library::checked_block_call_v1::<i32, _, _, _, ITEMS_PER_THREAD>(
                    direct_load_full,
                    (input,),
                    (values,),
                );
            }
        }
    }

    /// # Safety
    /// The payload is initialized. All threads participate with one block-wide
    /// output origin, valid count and partial selection, satisfying the exact CUB ABI.
    #[inline(never)]
    pub unsafe fn checked_store<const ITEMS_PER_THREAD: usize, const TRANSPOSE: bool>(
        output: *mut i32,
        values: *mut i32,
        valid: i32,
        partial: bool,
    ) {
        unsafe {
            if partial {
                if TRANSPOSE {
                    cuda_device::library::checked_block_call_v1::<i32, _, _, _, ITEMS_PER_THREAD>(
                        transpose_store_partial,
                        (output, valid),
                        (values,),
                    );
                } else {
                    cuda_device::library::checked_block_call_v1::<i32, _, _, _, ITEMS_PER_THREAD>(
                        direct_store_partial,
                        (output, valid),
                        (values,),
                    );
                }
            } else if TRANSPOSE {
                cuda_device::library::checked_block_call_v1::<i32, _, _, _, ITEMS_PER_THREAD>(
                    transpose_store_full,
                    (output,),
                    (values,),
                );
            } else {
                cuda_device::library::checked_block_call_v1::<i32, _, _, _, ITEMS_PER_THREAD>(
                    direct_store_full,
                    (output,),
                    (values,),
                );
            }
        }
    }
}
#[cfg(feature = "checked")]
pub use checked::{checked_load, checked_store};
