// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use crate::{
    Alignment, Block, GroupScope, Numeric, SupportedAlignment, ThreadData, ThreadGroup,
    ThreadsWithinWarp, Warp,
};
use cuda_device::library;

mod boundary;

/// Convert striped input into blocked output (the default Exchange mode).
///
/// For a group of `G` threads with `N` items each, output lane `r`, slot `i`
/// reads input lane `(r * N + i) % G`, slot `(r * N + i) / G`.
#[derive(Clone, Copy, Debug, Default)]
pub struct StripedToBlocked;

/// Convert blocked input into striped output.
///
/// For a group of `G` threads with `N` items each, output lane `r`, slot `i`
/// reads input lane `(i * G + r) / N`, slot `(i * G + r) % N`.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockedToStriped;

/// A sealed compile-time Exchange direction: [`StripedToBlocked`] or
/// [`BlockedToStriped`]. Neither direction changes the input payload.
pub trait ExchangeMode: sealed::Mode {}
impl sealed::Mode for StripedToBlocked {
    const KIND: u8 = 0;
}
impl sealed::Mode for BlockedToStriped {
    const KIND: u8 = 1;
}
impl ExchangeMode for StripedToBlocked {}
impl ExchangeMode for BlockedToStriped {}

/// A sealed group supporting Exchange: a complete [`Block`], physical [`Warp`],
/// or logical Warp of width 1, 2, 4, 8, 16 or 32, with either exhaustive selection.
///
/// Every selected member must participate. Sibling Warp groups may choose
/// different paths. Warp operations require an enclosing Block containing
/// complete physical warps. Block supports all three launch dimensions and uses
/// x-fastest thread rank. Non-power-of-two prefixes and partial Block groups
/// are unsupported; a descriptor alone does not prove participation.
pub trait ExchangeGroup: GroupScope + sealed::Group {}
impl ExchangeGroup for Block {}
impl sealed::Group for Block {
    const WARP: bool = false;
    #[inline(always)]
    unsafe fn call<T: Copy, R: Fn() -> &'static str + Copy, P: Copy, const I: usize>(
        registration: R,
        payload: P,
    ) {
        // SAFETY: the private boundary supplies live typed payloads and the
        // matching recipe. The marker checks participation and implicit scratch.
        unsafe {
            library::checked_block_payload_call_v1::<T, _, _, _, _, I>(
                registration,
                (),
                (),
                payload,
            )
        }
    }
}
macro_rules! warp_group {
    ($scope:ty, $width:literal $(, $e:ident)?) => {
        impl<$(const $e: bool)?> ExchangeGroup for $scope {}
        impl<$(const $e: bool)?> sealed::Group for $scope {
            const WARP: bool = true;
            #[inline(always)]
            unsafe fn call<T: Copy, R: Fn() -> &'static str + Copy, P: Copy, const I: usize>(
                registration: R, payload: P,
            ) {
                // SAFETY: the private boundary supplies the matching recipe and
                // live payloads; the genuine marker checks this canonical width.
                unsafe { library::checked_warp_payload_call_v1::<T, _, _, _, $width, I>(registration, (), payload) }
            }
        }
    };
}
warp_group!(Warp, 32);
warp_group!(ThreadsWithinWarp<1, E>, 1, E);
warp_group!(ThreadsWithinWarp<2, E>, 2, E);
warp_group!(ThreadsWithinWarp<4, E>, 4, E);
warp_group!(ThreadsWithinWarp<8, E>, 8, E);
warp_group!(ThreadsWithinWarp<16, E>, 16, E);
warp_group!(ThreadsWithinWarp<32, E>, 32, E);

/// Static direction for [`exchange`]. [`Default`] selects [`StripedToBlocked`].
///
/// `M` is an [`ExchangeMode`], selected by the consuming [`mode`](Self::mode)
/// builder. There are no count, algorithm, rank or explicit-storage options.
#[derive(Clone, Copy, Debug)]
pub struct ExchangeOptions<M = StripedToBlocked> {
    mode: M,
}
impl Default for ExchangeOptions {
    fn default() -> Self {
        Self {
            mode: StripedToBlocked,
        }
    }
}
impl<M: ExchangeMode> ExchangeOptions<M> {
    /// Select the compile-time layout conversion.
    ///
    /// ```
    /// use cuda_coop as coop;
    /// let _: coop::ExchangeOptions<coop::BlockedToStriped> =
    ///     coop::ExchangeOptions::default().mode(coop::BlockedToStriped);
    /// ```
    pub fn mode<M2: ExchangeMode>(self, mode: M2) -> ExchangeOptions<M2> {
        ExchangeOptions { mode }
    }
}

/// Rearrange a group's per-thread values into a fresh payload of the same type,
/// extent and alignment, preserving every input value and validity bit.
///
/// `group` selects an [`ExchangeGroup`]. `input` contains `ITEMS_PER_THREAD`
/// initialized slots per member, and `options` selects an [`ExchangeMode`].
/// The default converts striped input into blocked output. Blocked order gives
/// lane `r`, slot `i` logical index `r * ITEMS_PER_THREAD + i`; striped order
/// gives it `i * group.count() + r`. Warp conversions act independently in each
/// canonical group. All members use the same element type and item count.
///
/// `T` is inferred from input and supports all ten [`Numeric`] types.
/// `ITEMS_PER_THREAD` must be positive. `ALIGN` retains the input's supported
/// minimum alignment in the fresh result. Shared input and ordinary mutable
/// reborrows are accepted; scalars and owned-by-value payloads are unsupported.
/// Every returned slot is initialized, and changing the result cannot change
/// the input. The operation rearranges per-thread values; it does not load or store a
/// global-memory tile.
///
/// Every selected member must participate. Uninitialized input traps before
/// native numeric reads. Oxide plans implicit shared scratch using the actual
/// CUB layout and synchronizes before its reuse. This operation does not promise
/// global-memory completion: use the group's `sync()` for dependent global
/// accesses. Unsupported groups or modes fail compilation. Host code can check
/// these signatures; execution requires an Oxide-compiled CUDA kernel.
///
/// This same-extent signature needs no `generic_const_exprs` feature in callers.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn blocked<T: coop::Numeric, G: coop::ExchangeGroup, const ITEMS_PER_THREAD: usize>(
///     group: coop::ThreadGroup<G>, input: &coop::ThreadData<T, ITEMS_PER_THREAD>,
/// ) -> coop::ThreadData<T, ITEMS_PER_THREAD> {
///     coop::exchange(group, input, Default::default())
/// }
/// ```
///
/// A scalar is not a one-item payload:
/// ```compile_fail,E0308
/// use cuda_coop as coop;
/// fn invalid() {
///     coop::exchange(coop::this_block(), &3_i32, Default::default());
/// }
/// ```
///
/// Non-power-of-two Warp prefixes do not support Exchange:
/// ```compile_fail,E0277
/// use cuda_coop as coop;
/// fn invalid(values: &coop::ThreadData<i32, 4>) {
///     coop::exchange(coop::this_warp().group_by_non_exhaustive::<17>(), values,
///         Default::default());
/// }
/// ```
#[inline(always)]
pub fn exchange<T, G, M, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
    _group: ThreadGroup<G>,
    input: &ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
    options: ExchangeOptions<M>,
) -> ThreadData<T, ITEMS_PER_THREAD, ALIGN>
where
    T: Numeric,
    G: ExchangeGroup,
    M: ExchangeMode,
    Alignment<ALIGN>: SupportedAlignment,
{
    boundary::exchange::<T, G, M, ITEMS_PER_THREAD, ALIGN>(input, options)
}

mod sealed {
    pub trait Mode {
        const KIND: u8;
    }
    pub trait Group {
        const WARP: bool;
        unsafe fn call<T: Copy, R: Fn() -> &'static str + Copy, P: Copy, const I: usize>(
            registration: R,
            payload: P,
        );
    }
}
