// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use core::marker::PhantomData;
use cuda_device::{cluster, thread};

mod sealed {
    pub trait Scope {
        fn member() -> u8;
    }
    pub trait Level {}
    pub trait Integer {}
    pub trait Group {}
}

/// A supported physical or mapped group scope.
///
/// Sealed to this crate's scope types. A scope grants no collective
/// participation or synchronization guarantee.
pub trait GroupScope: sealed::Scope {}

/// A physical CUDA hierarchy level: Thread, Warp, Block, Cluster, or Grid.
///
/// This sealed trait supplies the level argument to [`ThreadGroup::rank_at`]
/// and [`ThreadGroup::count_at`]. Mapped scopes are not query levels.
pub trait Level: GroupScope + sealed::Level {}

macro_rules! levels {
    ($(($name:ident, $new:ident, $doc:literal)),+ $(,)?) => {$ (
        #[doc = $doc]
        #[derive(Clone, Copy, Debug)]
        pub enum $name {}
        impl sealed::Scope for $name {
            #[inline(always)] fn member() -> u8 { 1 }
        }
        impl GroupScope for $name {}
        impl sealed::Level for $name {}
        impl Level for $name {}
        #[doc = concat!("Describe the current ", $doc, "\n\nNo dimensions are supplied. This owns no resources and does not synchronize.")]
        #[inline(always)]
        pub const fn $new() -> ThreadGroup<$name> { ThreadGroup(PhantomData) }
    )+};
}
levels!(
    (Thread, this_thread, "CUDA thread."),
    (Warp, this_warp, "physical CUDA warp (32 lanes)."),
    (Block, this_block, "CUDA thread block."),
    (Cluster, this_cluster, "CUDA thread-block cluster."),
    (Grid, this_grid, "CUDA grid."),
);

/// Integer types accepted by rank/count conversions.
///
/// Sealed to `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, and `u64`.
/// Conversions follow integer `as` casts, including truncation and signed
/// reinterpretation. Choose a type wide enough for the result.
pub trait QueryInteger: sealed::Integer + Copy {
    #[doc(hidden)]
    fn from_u64(value: u64) -> Self;
    #[doc(hidden)]
    fn to_u64(self) -> u64;
}
macro_rules! integers {
    ($($t:ty),+) => {$ (
        impl sealed::Integer for $t {}
        impl QueryInteger for $t {
            #[inline(always)] fn from_u64(value: u64) -> Self { value as Self }
            #[inline(always)] fn to_u64(self) -> u64 { self as u64 }
        }
    )+};
}
integers!(i8, u8, i16, u16, i32, u32, i64, u64);

/// A supported scope/level query and its natural integer width.
///
/// Sealed to [`ThreadGroup`] implementations. Physical groups accept every
/// physical level. [`ThreadsWithinWarp`] accepts Thread/Warp;
/// [`WarpsWithinBlock`] accepts Thread/Warp/Block.
///
/// ```no_run
/// use cuda_coop::{GroupScope, QueryAt, Thread, ThreadGroup};
/// fn threads<S: GroupScope>(group: ThreadGroup<S>)
///     -> <ThreadGroup<S> as QueryAt<Thread>>::Output
/// where ThreadGroup<S>: QueryAt<Thread>
/// { group.count() }
/// ```
pub trait QueryAt<L: Level>: sealed::Group {
    /// `u64` when either scope or query level is Grid, otherwise `u32`.
    type Output: QueryInteger;
    #[doc(hidden)]
    fn query_rank(&self) -> Self::Output;
    #[doc(hidden)]
    fn query_count(&self) -> Self::Output;
}

/// A dimension-free descriptor of the executing thread's current group.
///
/// Construct with [`this_thread`], [`this_warp`], [`this_block`],
/// [`this_cluster`], or [`this_grid`]. `S` fixes the available queries.
/// Copies describe the thread executing each query, not the original thread.
///
/// Native queries require an Oxide-compiled kernel. They read launch state
/// without allocating, moving payloads, or synchronizing. Descriptors carry
/// no uniformity or collective safety proof. Cluster queries currently need
/// sm_90 or newer; an ordinary launch has one block per cluster.
#[derive(Debug)]
pub struct ThreadGroup<S: GroupScope>(PhantomData<S>);
impl<S: GroupScope> Copy for ThreadGroup<S> {}
impl<S: GroupScope> Clone for ThreadGroup<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: GroupScope> sealed::Group for ThreadGroup<S> {}
impl<S: GroupScope> ThreadGroup<S> {
    /// Return the thread's rank in this group; equivalent to
    /// `rank_at::<Thread>()`. Panics (traps on device) for mapped nonmembers.
    #[inline(always)]
    pub fn rank(self) -> <Self as QueryAt<Thread>>::Output
    where
        Self: QueryAt<Thread>,
    {
        self.rank_at::<Thread>()
    }

    /// Count threads in this group; equivalent to `count_at::<Thread>()`.
    /// A physical warp has 32 lanes, including a partially occupied last warp.
    #[inline(always)]
    pub fn count(self) -> <Self as QueryAt<Thread>>::Output
    where
        Self: QueryAt<Thread>,
    {
        self.count_at::<Thread>()
    }

    /// Query rank at `L`, with x-fastest linearization.
    ///
    /// An inner level selects a constituent's rank in this group; an outer
    /// level selects this group's rank in that parent. Same-level rank is
    /// zero. The result is `u64` if Grid is involved, otherwise `u32`.
    /// Panics (traps on device) for an excluded mapped-group member.
    ///
    /// ```no_run
    /// use cuda_coop as coop;
    /// let block = coop::this_block();
    /// let thread: u32 = block.rank();
    /// let warp: u32 = block.rank_at::<coop::Warp>();
    /// let block_in_grid: u64 = block.rank_at::<coop::Grid>();
    /// ```
    #[inline(always)]
    pub fn rank_at<L: Level>(self) -> <Self as QueryAt<L>>::Output
    where
        Self: QueryAt<L>,
    {
        self.query_rank()
    }

    /// Count constituents or parents at `L`.
    ///
    /// An inner level counts constituents in this group; an outer level
    /// counts these groups in that parent. Same-level count is one. The
    /// result is `u64` if Grid is involved, otherwise `u32`. Mapped counts
    /// remain defined for excluded trailing threads.
    #[inline(always)]
    pub fn count_at<L: Level>(self) -> <Self as QueryAt<L>>::Output
    where
        Self: QueryAt<L>,
    {
        self.query_count()
    }

    /// Convert [`Self::rank`] to integer `I` with integer `as` semantics.
    /// Mapped membership checks still apply.
    #[inline(always)]
    pub fn rank_as<I: QueryInteger>(self) -> I
    where
        Self: QueryAt<Thread>,
    {
        self.rank_at_as::<I, Thread>()
    }

    /// Convert [`Self::count`] to integer `I` with integer `as` semantics.
    #[inline(always)]
    pub fn count_as<I: QueryInteger>(self) -> I
    where
        Self: QueryAt<Thread>,
    {
        self.count_at_as::<I, Thread>()
    }

    /// Convert [`Self::rank_at`] at `L` to integer `I` with integer `as`
    /// semantics. Mapped membership checks still apply.
    #[inline(always)]
    pub fn rank_at_as<I: QueryInteger, L: Level>(self) -> I
    where
        Self: QueryAt<L>,
    {
        I::from_u64(self.rank_at::<L>().to_u64())
    }

    /// Convert [`Self::count_at`] at `L` to integer `I` with integer `as`
    /// semantics.
    #[inline(always)]
    pub fn count_at_as<I: QueryInteger, L: Level>(self) -> I
    where
        Self: QueryAt<L>,
    {
        I::from_u64(self.count_at::<L>().to_u64())
    }

    /// Return `1u8` for a member, `0u8` for an excluded trailing unit.
    /// Physical groups return one. This query grants no participation proof.
    #[inline(always)]
    pub fn is_member(self) -> u8 {
        S::member()
    }
}

/// A dimension-free view of the current physical CUDA hierarchy.
///
/// [`Self::new`] and [`Self::current`] are equivalent. The hierarchy stores
/// no launch dimensions or resources and grants no synchronization capability.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThreadHierarchy(());
/// Alias for [`ThreadHierarchy`].
pub type Hierarchy = ThreadHierarchy;
impl ThreadHierarchy {
    /// Describe the current launch without supplying dimensions.
    pub const fn new() -> Self {
        Self(())
    }
    /// Describe the current launch; equivalent to [`Self::new`].
    pub const fn current() -> Self {
        Self::new()
    }
    /// Describe the current thread.
    pub const fn thread(self) -> ThreadGroup<Thread> {
        this_thread()
    }
    /// Describe the current physical warp.
    pub const fn warp(self) -> ThreadGroup<Warp> {
        this_warp()
    }
    /// Describe the current thread block.
    pub const fn block(self) -> ThreadGroup<Block> {
        this_block()
    }
    /// Describe the current thread-block cluster.
    pub const fn cluster(self) -> ThreadGroup<Cluster> {
        this_cluster()
    }
    /// Describe the current grid.
    pub const fn grid(self) -> ThreadGroup<Grid> {
        this_grid()
    }
}

// Read only the registers required by the monomorphized query.
#[inline(always)]
fn thread_rank_block() -> u32 {
    (thread::threadIdx_z() * thread::blockDim_y() + thread::threadIdx_y()) * thread::blockDim_x()
        + thread::threadIdx_x()
}
#[inline(always)]
fn threads_block() -> u32 {
    thread::blockDim_x() * thread::blockDim_y() * thread::blockDim_z()
}
#[inline(always)]
fn warps_block() -> u32 {
    threads_block().div_ceil(32)
}
#[inline(always)]
fn linear_u64(x: u32, y: u32, z: u32, nx: u32, ny: u32) -> u64 {
    (u64::from(z) * u64::from(ny) + u64::from(y)) * u64::from(nx) + u64::from(x)
}
#[inline(always)]
fn block_rank_grid() -> u64 {
    linear_u64(
        thread::blockIdx_x(),
        thread::blockIdx_y(),
        thread::blockIdx_z(),
        thread::gridDim_x(),
        thread::gridDim_y(),
    )
}
#[inline(always)]
fn blocks_grid() -> u64 {
    u64::from(thread::gridDim_x()) * u64::from(thread::gridDim_y()) * u64::from(thread::gridDim_z())
}
#[inline(always)]
fn cluster_rank_grid() -> u64 {
    linear_u64(
        thread::blockIdx_x() / cluster::cluster_nctaidX(),
        thread::blockIdx_y() / cluster::cluster_nctaidY(),
        thread::blockIdx_z() / cluster::cluster_nctaidZ(),
        thread::gridDim_x() / cluster::cluster_nctaidX(),
        thread::gridDim_y() / cluster::cluster_nctaidY(),
    )
}
#[inline(always)]
fn clusters_grid() -> u64 {
    u64::from(thread::gridDim_x() / cluster::cluster_nctaidX())
        * u64::from(thread::gridDim_y() / cluster::cluster_nctaidY())
        * u64::from(thread::gridDim_z() / cluster::cluster_nctaidZ())
}
macro_rules! query {
    ($scope:ty, $level:ty, $out:ty, $rank:expr, $count:expr) => {
        impl QueryAt<$level> for ThreadGroup<$scope> {
            type Output = $out;
            #[inline(always)]
            fn query_rank(&self) -> $out {
                $rank
            }
            #[inline(always)]
            fn query_count(&self) -> $out {
                $count
            }
        }
    };
}
macro_rules! pair {
    ($inner:ty, $outer:ty, $out:ty, $rank:expr, $count:expr) => {
        query!($inner, $outer, $out, $rank, $count);
        query!($outer, $inner, $out, $rank, $count);
    };
}
query!(Thread, Thread, u32, 0, 1);
query!(Warp, Warp, u32, 0, 1);
query!(Block, Block, u32, 0, 1);
query!(Cluster, Cluster, u32, 0, 1);
query!(Grid, Grid, u64, 0, 1);
pair!(Thread, Warp, u32, thread_rank_block() % 32, 32);
pair!(Thread, Block, u32, thread_rank_block(), threads_block());
pair!(Warp, Block, u32, thread_rank_block() / 32, warps_block());
pair!(
    Thread,
    Cluster,
    u32,
    cluster::block_rank() * threads_block() + thread_rank_block(),
    cluster::cluster_size() * threads_block()
);
pair!(
    Warp,
    Cluster,
    u32,
    cluster::block_rank() * warps_block() + thread_rank_block() / 32,
    cluster::cluster_size() * warps_block()
);
pair!(
    Block,
    Cluster,
    u32,
    cluster::block_rank(),
    cluster::cluster_size()
);
pair!(
    Thread,
    Grid,
    u64,
    block_rank_grid()
        .wrapping_mul(u64::from(threads_block()))
        .wrapping_add(u64::from(thread_rank_block())),
    blocks_grid().wrapping_mul(u64::from(threads_block()))
);
pair!(
    Warp,
    Grid,
    u64,
    block_rank_grid()
        .wrapping_mul(u64::from(warps_block()))
        .wrapping_add(u64::from(thread_rank_block() / 32)),
    blocks_grid().wrapping_mul(u64::from(warps_block()))
);
pair!(Block, Grid, u64, block_rank_grid(), blocks_grid());
pair!(Cluster, Grid, u64, cluster_rank_grid(), clusters_grid());

/// Consecutive groups of `N` lanes within a physical warp.
///
/// Construct with [`ThreadGroup<Warp>::group_by`] or its non-exhaustive
/// variant. `N` is in 1..=32. `EXHAUSTIVE = true` requires divisibility;
/// otherwise trailing lanes are excluded. The block must have complete
/// warps. Only Thread/Warp queries are available; groups cannot be nested.
///
/// ```compile_fail,E0277
/// let group = cuda_coop::this_warp().group_by::<8>();
/// group.rank_at::<cuda_coop::Block>();
/// ```
/// ```compile_fail,E0599
/// cuda_coop::this_warp().group_by::<8>().group_by::<2>();
/// ```
#[derive(Clone, Copy, Debug)]
pub enum ThreadsWithinWarp<const N: usize, const EXHAUSTIVE: bool = true> {}
/// Consecutive groups of `N` physical warps within a block.
///
/// Construct with [`ThreadGroup<Block>::group_by`] or its non-exhaustive
/// variant. The block must have complete warps and at least `N` warps.
/// `EXHAUSTIVE = true` requires divisibility; otherwise trailing warps are
/// excluded. Only Thread/Warp/Block queries are available; no nesting.
///
/// ```compile_fail,E0277
/// cuda_coop::this_block().group_by::<2>().count_at::<cuda_coop::Cluster>();
/// ```
#[derive(Clone, Copy, Debug)]
pub enum WarpsWithinBlock<const N: usize, const EXHAUSTIVE: bool = true> {}
impl<const N: usize, const E: bool> ThreadsWithinWarp<N, E> {
    const VALID: () = {
        assert!(N > 0 && N <= 32, "lane partition size must be in 1..=32");
        assert!(
            !E || 32 % N == 0,
            "exhaustive lane partition must divide 32"
        );
    };
}
impl<const N: usize, const E: bool> WarpsWithinBlock<N, E> {
    const VALID: () = {
        assert!(N > 0 && N <= 32, "warp partition size must be in 1..=32");
    };
}
impl<const N: usize, const E: bool> sealed::Scope for ThreadsWithinWarp<N, E> {
    #[inline(always)]
    fn member() -> u8 {
        ((thread_rank_block() % 32) < (32 / N as u32) * N as u32) as u8
    }
}
impl<const N: usize, const E: bool> GroupScope for ThreadsWithinWarp<N, E> {}
impl<const N: usize, const E: bool> sealed::Scope for WarpsWithinBlock<N, E> {
    #[inline(always)]
    fn member() -> u8 {
        ((thread_rank_block() / 32) < (warps_block() / N as u32) * N as u32) as u8
    }
}
impl<const N: usize, const E: bool> GroupScope for WarpsWithinBlock<N, E> {}
impl ThreadGroup<Warp> {
    /// Partition into consecutive groups of `N` lanes.
    ///
    /// `N` must be positive and divide 32 (checked at compilation). Panics
    /// (traps on device) if the block has a partial physical warp.
    ///
    /// ```compile_fail,E0080
    /// cuda_coop::this_warp().group_by::<0>();
    /// ```
    /// ```compile_fail,E0080
    /// cuda_coop::this_warp().group_by::<6>();
    /// ```
    #[inline(always)]
    pub fn group_by<const N: usize>(self) -> ThreadGroup<ThreadsWithinWarp<N>> {
        self.partition::<N, true>()
    }
    /// Partition into `N`-lane groups, excluding leftover lanes.
    ///
    /// `N` must be in 1..=32. Panics (traps on device) for a partial physical
    /// warp. Check [`ThreadGroup::is_member`] before querying a member rank.
    #[inline(always)]
    pub fn group_by_non_exhaustive<const N: usize>(
        self,
    ) -> ThreadGroup<ThreadsWithinWarp<N, false>> {
        self.partition::<N, false>()
    }
    #[inline(always)]
    fn partition<const N: usize, const E: bool>(self) -> ThreadGroup<ThreadsWithinWarp<N, E>> {
        let () = ThreadsWithinWarp::<N, E>::VALID;
        assert!(
            threads_block().is_multiple_of(32),
            "lane partitions require complete physical warps"
        );
        ThreadGroup(PhantomData)
    }
}
impl ThreadGroup<Block> {
    /// Partition into consecutive groups of `N` physical warps.
    ///
    /// `N` must be in 1..=32 (checked at compilation). Panics (traps on
    /// device) unless the block has complete warps and its warp count is
    /// divisible by `N`.
    ///
    /// ```compile_fail,E0080
    /// cuda_coop::this_block().group_by::<33>();
    /// ```
    #[inline(always)]
    pub fn group_by<const N: usize>(self) -> ThreadGroup<WarpsWithinBlock<N>> {
        self.partition::<N, true>()
    }
    /// Partition into `N`-warp groups, excluding leftover warps.
    ///
    /// `N` must be in 1..=32. Panics (traps on device) unless the block has
    /// complete warps and at least `N` warps. Check [`ThreadGroup::is_member`]
    /// before querying a possibly excluded thread's rank.
    #[inline(always)]
    pub fn group_by_non_exhaustive<const N: usize>(
        self,
    ) -> ThreadGroup<WarpsWithinBlock<N, false>> {
        self.partition::<N, false>()
    }
    #[inline(always)]
    fn partition<const N: usize, const E: bool>(self) -> ThreadGroup<WarpsWithinBlock<N, E>> {
        let () = WarpsWithinBlock::<N, E>::VALID;
        let warps = warps_block();
        assert!(
            threads_block().is_multiple_of(32),
            "warp partitions require complete physical warps"
        );
        assert!(
            N as u32 <= warps,
            "partition exceeds the block's warp count"
        );
        assert!(
            !E || warps.is_multiple_of(N as u32),
            "exhaustive partition must divide the block's warp count"
        );
        ThreadGroup(PhantomData)
    }
}
macro_rules! mapped_query {
    ($scope:ident, $level:ty, $rank:expr, $count:expr) => {
        impl<const N: usize, const E: bool> QueryAt<$level> for ThreadGroup<$scope<N, E>> {
            type Output = u32;
            #[inline(always)]
            fn query_rank(&self) -> u32 {
                assert!(
                    self.is_member() == 1,
                    "excluded thread has no mapped-group rank"
                );
                $rank
            }
            #[inline(always)]
            fn query_count(&self) -> u32 {
                $count
            }
        }
    };
}
mapped_query!(
    ThreadsWithinWarp,
    Thread,
    thread_rank_block() % 32 % N as u32,
    N as u32
);
mapped_query!(
    ThreadsWithinWarp,
    Warp,
    thread_rank_block() % 32 / N as u32,
    32 / N as u32
);
mapped_query!(
    WarpsWithinBlock,
    Thread,
    (thread_rank_block() / 32 % N as u32) * 32 + thread_rank_block() % 32,
    N as u32 * 32
);
mapped_query!(
    WarpsWithinBlock,
    Warp,
    thread_rank_block() / 32 % N as u32,
    N as u32
);
mapped_query!(
    WarpsWithinBlock,
    Block,
    thread_rank_block() / 32 / N as u32,
    warps_block() / N as u32
);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grid_linearization_promotes_before_multiplication() {
        assert_eq!(
            linear_u64(7, 65534, 2, 2_147_483_647, 65535),
            422_203_874_934_795
        );
    }
    #[test]
    fn query_casts_preserve_integer_cast_semantics() {
        assert_eq!(<i8 as QueryInteger>::from_u64(255), -1);
        assert_eq!(<u16 as QueryInteger>::from_u64(65537), 1);
        assert_eq!(
            <u64 as QueryInteger>::from_u64(u32::MAX as u64 + 1),
            4_294_967_296
        );
    }
}
