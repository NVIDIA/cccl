// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for the low-level top-k rank API in `cub/cub/block/block_topk_rank.cuh`
// (`block_topk_sieve::select_max/min`, `refine_max/min`, and
// `block_topk_rank::rank_key_states`).
//
// `rank_key_states` is non-deterministic in both selected-subset and output
// order when ties remain. All checks compare the scattered output as a
// value-multiset against a host reference; the fixtures are shaped so that
// multiset is uniquely determined regardless of how the ranker resolves the
// boundary ties (see `gen_keys_from_boundary_key` and the multi-key test).

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_topk_rank.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/type_traits>

#include <algorithm>

#include "catch2_test_block_topk_common.cuh"
#include <c2h/catch2_test_helper.h>

namespace
{
// --- Kernels under test ---

// `select_*` + `rank_key_states`, then both round-trip the per-thread keys
// back to `g_keys_out` (must be byte-identical to `g_in`) and scatter the
// selected keys to `g_top[0, k)`. `k` is the size of `g_top`.
template <typename KeyT, int BlockDim, int ItemsPerThread, bool IsFullTile, bool BlockedInput, bool SelectMax>
__global__ void single_key_kernel(
  cuda::std::span<const KeyT> g_in,
  cuda::std::span<KeyT> g_keys_out,
  cuda::std::span<KeyT> g_top,
  cuda::std::span<bool, 1> g_has_ties)
{
  using sieve_t = cub::detail::block_topk_sieve<KeyT, BlockDim>;
  using rank_t  = cub::detail::block_topk_rank<BlockDim>;

  const int k = static_cast<int>(g_top.size());

  __shared__ union
  {
    typename sieve_t::TempStorage sieve;
    typename rank_t::TempStorage rank;
  } smem;

  // OOB lanes seeded with the most-attractive value so any escape into
  // the candidate set displaces a real winner and the compare flags it.
  KeyT keys[ItemsPerThread];
  constexpr KeyT oob_sentinel =
    SelectMax ? cuda::std::numeric_limits<KeyT>::max() : cuda::std::numeric_limits<KeyT>::lowest();
  if constexpr (BlockedInput)
  {
    cub::LoadDirectBlocked(
      static_cast<int>(threadIdx.x), g_in.data(), keys, static_cast<int>(g_in.size()), oob_sentinel);
  }
  else
  {
    cub::LoadDirectStriped<BlockDim>(
      static_cast<int>(threadIdx.x), g_in.data(), keys, static_cast<int>(g_in.size()), oob_sentinel);
  }

  sieve_t sieve(smem.sieve);
  auto states = [&] {
    if constexpr (SelectMax)
    {
      return sieve.template select_max<IsFullTile, BlockedInput>(keys, k, static_cast<int>(g_in.size()));
    }
    else
    {
      return sieve.template select_min<IsFullTile, BlockedInput>(keys, k, static_cast<int>(g_in.size()));
    }
  }();
  __syncthreads();

  if (threadIdx.x == 0)
  {
    g_has_ties[0] = states.has_ties();
  }

  int ranks[ItemsPerThread];
  rank_t(smem.rank).rank_key_states(states, ranks);

  for (int i = 0; i < ItemsPerThread; ++i)
  {
    if (ranks[i] >= 0)
    {
      g_top[ranks[i]] = keys[i];
    }
  }

  // Round-trip the keys: bit-identical to the input. Bounded store skips OOB.
  if constexpr (BlockedInput)
  {
    cub::StoreDirectBlocked(static_cast<int>(threadIdx.x), g_keys_out.data(), keys, static_cast<int>(g_keys_out.size()));
  }
  else
  {
    cub::StoreDirectStriped<BlockDim>(
      static_cast<int>(threadIdx.x), g_keys_out.data(), keys, static_cast<int>(g_keys_out.size()));
  }
}

// Primary `select_max` + (if ties) secondary `refine_max`, then
// `rank_key_states`. `g_has_ties[0]` / `g_has_ties[1]` snapshot
// `has_ties()` before and after the (possibly skipped) refine.
template <typename PrimaryT, typename SecondaryT, int BlockDim, int ItemsPerThread, bool IsFullTile, bool SelectMax>
__global__ void multi_key_kernel(
  cuda::std::span<const PrimaryT> g_primary,
  cuda::std::span<const SecondaryT> g_secondary,
  cuda::std::span<PrimaryT> g_top_primary,
  cuda::std::span<SecondaryT> g_top_secondary,
  cuda::std::span<bool, 2> g_has_ties)
{
  using primary_sieve_t   = cub::detail::block_topk_sieve<PrimaryT, BlockDim>;
  using secondary_sieve_t = cub::detail::block_topk_sieve<SecondaryT, BlockDim>;
  using rank_t            = cub::detail::block_topk_rank<BlockDim>;

  const int k = static_cast<int>(g_top_primary.size());

  __shared__ union
  {
    typename primary_sieve_t::TempStorage primary_sieve;
    typename secondary_sieve_t::TempStorage secondary_sieve;
    typename rank_t::TempStorage rank;
  } smem;

  // OOB sentinels: same convention as `single_key_kernel`.
  constexpr PrimaryT primary_oob_sentinel =
    SelectMax ? cuda::std::numeric_limits<PrimaryT>::max() : cuda::std::numeric_limits<PrimaryT>::lowest();
  constexpr SecondaryT secondary_oob_sentinel =
    SelectMax ? cuda::std::numeric_limits<SecondaryT>::max() : cuda::std::numeric_limits<SecondaryT>::lowest();
  PrimaryT primary_keys[ItemsPerThread];
  cub::LoadDirectBlocked(
    static_cast<int>(threadIdx.x),
    g_primary.data(),
    primary_keys,
    static_cast<int>(g_primary.size()),
    primary_oob_sentinel);
  // Loaded unconditionally so they're in scope for the final scatter.
  SecondaryT secondary_keys[ItemsPerThread];
  cub::LoadDirectBlocked(
    static_cast<int>(threadIdx.x),
    g_secondary.data(),
    secondary_keys,
    static_cast<int>(g_secondary.size()),
    secondary_oob_sentinel);

  primary_sieve_t primary_sieve(smem.primary_sieve);
  constexpr bool blocked_input = true;
  auto states                  = [&] {
    if constexpr (SelectMax)
    {
      return primary_sieve.template select_max<IsFullTile, blocked_input>(
        primary_keys, k, static_cast<int>(g_primary.size()));
    }
    else
    {
      return primary_sieve.template select_min<IsFullTile, blocked_input>(
        primary_keys, k, static_cast<int>(g_primary.size()));
    }
  }();
  __syncthreads();

  if (threadIdx.x == 0)
  {
    g_has_ties[0] = states.has_ties();
  }

  if (states.has_ties())
  {
    secondary_sieve_t secondary_sieve(smem.secondary_sieve);
    if constexpr (SelectMax)
    {
      secondary_sieve.refine_max(secondary_keys, states);
    }
    else
    {
      secondary_sieve.refine_min(secondary_keys, states);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    g_has_ties[1] = states.has_ties();
  }

  int ranks[ItemsPerThread];
  rank_t(smem.rank).rank_key_states(states, ranks);

  for (int i = 0; i < ItemsPerThread; ++i)
  {
    if (ranks[i] >= 0)
    {
      g_top_primary[ranks[i]]   = primary_keys[i];
      g_top_secondary[ranks[i]] = secondary_keys[i];
    }
  }
}

// `select_max` over a high bit window, then `refine_max` down through
// the lower windows. `g_last_hi` is the loop's `hi` cursor after
// termination -- `0` if every window ran, otherwise the lower bound of
// the last window before the `!has_ties()` early exit.
template <typename KeyT, int BlockDim, int ItemsPerThread, int WindowBits>
__global__ void
bit_window_kernel(cuda::std::span<const KeyT> g_in, cuda::std::span<KeyT> g_top, cuda::std::span<int, 1> g_last_hi)
{
  using sieve_t = cub::detail::block_topk_sieve<KeyT, BlockDim>;
  using rank_t  = cub::detail::block_topk_rank<BlockDim>;

  static_assert(int(sizeof(KeyT) * 8) % WindowBits == 0, "test currently requires window-aligned key width");

  constexpr int tile_size = BlockDim * ItemsPerThread;
  const int k             = static_cast<int>(g_top.size());

  __shared__ union
  {
    typename sieve_t::TempStorage sieve;
    typename rank_t::TempStorage rank;
  } smem;

  KeyT keys[ItemsPerThread];
  cub::LoadDirectBlocked(static_cast<int>(threadIdx.x), g_in.data(), keys);

  sieve_t sieve(smem.sieve);

  constexpr bool is_full_tile = true;
  int hi                      = int(sizeof(KeyT) * 8);
  auto states                 = sieve.template select_max<is_full_tile>(keys, k, tile_size, hi - WindowBits, hi);
  hi -= WindowBits;

  while (hi > 0)
  {
    __syncthreads();
    if (!states.has_ties())
    {
      break;
    }
    sieve.refine_max(keys, states, hi - WindowBits, hi);
    hi -= WindowBits;
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    g_last_hi[0] = hi;
  }

  int ranks[ItemsPerThread];
  rank_t(smem.rank).rank_key_states(states, ranks);

  for (int i = 0; i < ItemsPerThread; ++i)
  {
    if (ranks[i] >= 0)
    {
      g_top[ranks[i]] = keys[i];
    }
  }
}
} // namespace

// --- Type-list axes for the parameterized tests ---

// Direction axis: `false_type` = select_min, `true_type` = select_max.
using select_direction_max = c2h::type_list<cuda::std::false_type, cuda::std::true_type>;

using key_types = c2h::type_list<cuda::std::uint16_t, cuda::std::int32_t, cuda::std::uint64_t, float, double>;

using fp_key_types = c2h::type_list<float, double>;

// (BlockDim, ItemsPerThread) shape pair, used as a Catch2 type-list axis.
template <int BlockDim, int ItemsPerThread>
struct block_shape
{
  static constexpr int threads_per_block = BlockDim;
  static constexpr int items_per_thread  = ItemsPerThread;
};

// Four full-tile shapes sharing `tile_size == 512` so the per-shape `k`
// sweep is comparable across them.
using block_shapes_full_tile =
  c2h::type_list<block_shape<64, 8>, block_shape<256, 2>, block_shape<32, 16>, block_shape<128, 4>>;

// --- Test driver ---

// Runs `single_key_kernel` over `h_in` and checks bit-identity (round-trip
// output equals input) and selection correctness (first `min(k, num_valid)`
// slots of `g_top` match `h_ref` as a multiset). `k > num_valid` is
// supported. `h_ref` must be sized `min(k, num_valid)` (see
// `sorted_top_k`). Returns the kernel's `has_ties`.
template <typename KeyT, int BlockDim, int ItemsPerThread, bool IsFullTile, bool BlockedInput, bool SelectMax>
bool check_single_key(const c2h::host_vector<KeyT>& h_in, cuda::std::span<const KeyT> h_ref, int k)
{
  constexpr int tile  = BlockDim * ItemsPerThread;
  const int num_valid = static_cast<int>(h_in.size());
  REQUIRE(0 < k);
  REQUIRE(num_valid <= tile);
  REQUIRE((!IsFullTile || num_valid == tile));

  const int top_size = cuda::std::min(k, num_valid);
  REQUIRE(static_cast<int>(h_ref.size()) == top_size);

  c2h::device_vector<KeyT> d_in(h_in);
  c2h::device_vector<KeyT> d_keys_out(num_valid, KeyT{});
  // Tail (when `k > num_valid`) unused by the kernel, dropped pre-compare.
  c2h::device_vector<KeyT> d_top(k, KeyT{});
  c2h::device_vector<bool> d_has_ties(1, false);

  single_key_kernel<KeyT, BlockDim, ItemsPerThread, IsFullTile, BlockedInput, SelectMax><<<1, BlockDim>>>(
    to_span(d_in),
    to_span(d_keys_out),
    to_span(d_top),
    cuda::std::span<bool, 1>{cuda::std::to_address(d_has_ties.data()), 1});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_keys_out(d_keys_out);
  REQUIRE(bit_repr(h_in) == bit_repr(h_keys_out));

  const bool has_ties = c2h::host_vector<bool>(d_has_ties)[0];

  c2h::host_vector<KeyT> h_top(d_top);
  h_top.resize(top_size);
  std::sort(h_top.begin(), h_top.end(), comparator_t<SelectMax>{});
  c2h::host_vector<KeyT> h_ref_vec(h_ref.begin(), h_ref.end());
  // FP `==` already excuses the sieve's `+/-0.0` ambiguity; CAPTURE bit
  // patterns since the default printer rounds floats.
  CAPTURE(bit_repr(h_top), bit_repr(h_ref_vec));
  REQUIRE(h_top == h_ref_vec);

  return has_ties;
}

// ---------------------------------------------------------------------------
// 1) Random-data smoke test using `random_keys_centered`. Non-power-of-two
//    `(96, 5)` shape distinct from the other tests. Expected `has_ties` is
//    read off the sorted reference (`sorted[k-1] == sorted[k]`).
// ---------------------------------------------------------------------------

C2H_TEST("block_topk_sieve correctly selects on plain random inputs",
         "[block][topk][rank][smoke]",
         key_types,
         select_direction_max)
{
  using key_t                            = c2h::get<0, TestType>;
  static constexpr bool select_max       = c2h::get<1, TestType>::value;
  static constexpr int threads_per_block = 96;
  static constexpr int items_per_thread  = 5;
  static constexpr int tile_size         = threads_per_block * items_per_thread;

  rng_t rng(static_cast<cuda::std::uint32_t>(C2H_SEED(2).get()));

  c2h::host_vector<key_t> h_in = random_keys_centered<key_t>(tile_size, rng);

  const int k = GENERATE_COPY(values<int>({1, tile_size / 4, tile_size / 2, tile_size - 1}));
  CAPTURE(c2h::type_name<key_t>(), select_max, k);

  static constexpr bool is_full_tile  = true;
  static constexpr bool blocked_input = true;
  // Partial-sort to `k + 1` so one vector serves both the top-k check
  // (first `k`) and the boundary tie check below.
  const auto h_ref_kp1 = sorted_top_k<select_max>(h_in, k + 1);
  const bool has_ties =
    check_single_key<key_t, threads_per_block, items_per_thread, is_full_tile, blocked_input, select_max>(
      h_in, to_span(h_ref_kp1).first(k), k);

  const bool expected_has_ties = (static_cast<int>(h_ref_kp1.size()) > k) && (h_ref_kp1[k - 1] == h_ref_kp1[k]);
  CAPTURE(expected_has_ties);
  REQUIRE(has_ties == expected_has_ties);
}

// ---------------------------------------------------------------------------
// 2) Round-trip + selection on FP edge-case inputs. Only test pinning
//    `+0.0`, `-0.0`, `+/-inf` together in one input; non-FP coverage of
//    the "bit-distinct + has_ties == false" property lives in test 6.
// ---------------------------------------------------------------------------

C2H_TEST("block_topk_sieve preserves keys bit-faithfully across FP edge cases",
         "[block][topk][rank]",
         fp_key_types,
         select_direction_max)
{
  using key_t                            = c2h::get<0, TestType>;
  static constexpr bool select_max       = c2h::get<1, TestType>::value;
  static constexpr int threads_per_block = 128;
  static constexpr int items_per_thread  = 4;
  static constexpr int tile_size         = threads_per_block * items_per_thread;
  static_assert(tile_size >= 4, "FP edge-case poke needs at least 4 slots in h_in");

  rng_t rng(static_cast<cuda::std::uint32_t>(C2H_SEED(1).get()));
  c2h::host_vector<key_t> h_in = distinct_keys<key_t>(tile_size, rng);
  h_in[0]                      = static_cast<key_t>(-0.0);
  h_in[1]                      = static_cast<key_t>(+0.0);
  h_in[2]                      = cuda::std::numeric_limits<key_t>::infinity();
  h_in[3]                      = -cuda::std::numeric_limits<key_t>::infinity();
  thrust::shuffle(h_in.begin(), h_in.end(), rng);

  CAPTURE(c2h::type_name<key_t>(), select_max);

  // Partial-tile sections use a `num_valid` prefix. All-distinct bit patterns ->
  // `has_ties()` must end false.
  const int num_valid = tile_size / 2 + 7;
  c2h::host_vector<key_t> h_in_partial(h_in.begin(), h_in.begin() + num_valid);

  SECTION("full tile, blocked input")
  {
    static constexpr bool is_full_tile  = true;
    static constexpr bool blocked_input = true;
    const int k                         = GENERATE_COPY(values<int>({1, tile_size / 4, tile_size - 1}));
    CAPTURE(k);
    const auto h_ref = sorted_top_k<select_max>(h_in, k);
    const bool has_ties =
      check_single_key<key_t, threads_per_block, items_per_thread, is_full_tile, blocked_input, select_max>(
        h_in, to_span(h_ref), k);
    REQUIRE_FALSE(has_ties);
  }

  SECTION("partial tile, blocked input")
  {
    static constexpr bool is_full_tile  = false;
    static constexpr bool blocked_input = true;
    const int k                         = GENERATE_COPY(values<int>({1, num_valid / 4, num_valid - 1}));
    CAPTURE(num_valid, k);
    const auto h_ref = sorted_top_k<select_max>(h_in_partial, k);
    const bool has_ties =
      check_single_key<key_t, threads_per_block, items_per_thread, is_full_tile, blocked_input, select_max>(
        h_in_partial, to_span(h_ref), k);
    REQUIRE_FALSE(has_ties);
  }

  SECTION("partial tile, striped input")
  {
    static constexpr bool is_full_tile  = false;
    static constexpr bool blocked_input = false;
    const int k                         = GENERATE_COPY(values<int>({1, num_valid / 4, num_valid - 1}));
    CAPTURE(num_valid, k);
    const auto h_ref = sorted_top_k<select_max>(h_in_partial, k);
    const bool has_ties =
      check_single_key<key_t, threads_per_block, items_per_thread, is_full_tile, blocked_input, select_max>(
        h_in_partial, to_span(h_ref), k);
    REQUIRE_FALSE(has_ties);
  }
}

// ---------------------------------------------------------------------------
// 3) Single-key correctness on a full tile, swept over `block_shapes_full_tile`.
// ---------------------------------------------------------------------------

C2H_TEST("block_topk_sieve::select_* selects the right top-k on a full tile",
         "[block][topk][rank]",
         key_types,
         select_direction_max,
         block_shapes_full_tile)
{
  using key_t                            = c2h::get<0, TestType>;
  static constexpr bool select_max       = c2h::get<1, TestType>::value;
  using shape_t                          = c2h::get<2, TestType>;
  static constexpr int threads_per_block = shape_t::threads_per_block;
  static constexpr int items_per_thread  = shape_t::items_per_thread;
  static constexpr int tile_size         = threads_per_block * items_per_thread;
  static constexpr bool is_full_tile     = true;
  static constexpr bool blocked_input    = true;

  const int k = GENERATE_COPY(values<int>({1, 13, items_per_thread, threads_per_block, tile_size - 1, tile_size}));
  // 0 (order only) -> `tile_size - k` (every boundary copy overflows);
  // trailing entries go negative when `k > tile_size` but get narrowed off.
  const int overhang = GENERATE_COPY(overhang_generator(k == tile_size, {0, 1, (tile_size - k) / 2, tile_size - k}));

  auto run_check = [&](rng_t& rng, key_t boundary_key) {
    CAPTURE(c2h::type_name<key_t>(), select_max, k, overhang, boundary_key);
    c2h::host_vector<key_t> h_in = gen_keys_from_boundary_key<select_max>(tile_size, k, overhang, boundary_key, rng);
    const auto h_ref             = sorted_top_k<select_max>(h_in, k);
    const bool has_ties =
      check_single_key<key_t, threads_per_block, items_per_thread, is_full_tile, blocked_input, select_max>(
        h_in, to_span(h_ref), k);
    REQUIRE(has_ties == (overhang > 0));
  };

  // Split fixed/random so the seed loop only fans out on the random draw.
  SECTION("fixed boundary_key")
  {
    rng_t rng(static_cast<cuda::std::uint32_t>(C2H_SEED(1).get()));
    const key_t boundary_key = GENERATE_COPY(boundary_key_generator<key_t>());
    run_check(rng, boundary_key);
  }
  SECTION("random boundary_key")
  {
    rng_t rng(static_cast<cuda::std::uint32_t>(C2H_SEED(2).get()));
    const key_t boundary_key = random_boundary_key<key_t>(rng);
    run_check(rng, boundary_key);
  }
}

// ---------------------------------------------------------------------------
// 4) Single-key correctness on partial tiles, blocked and striped inputs.
// ---------------------------------------------------------------------------

C2H_TEST("block_topk_sieve handles partial tiles correctly", "[block][topk][rank]", key_types, select_direction_max)
{
  using key_t                      = c2h::get<0, TestType>;
  static constexpr bool select_max = c2h::get<1, TestType>::value;
  // Larger block, fewer items-per-thread than tests 2 and 3.
  static constexpr int threads_per_block = 256;
  static constexpr int items_per_thread  = 2;
  static constexpr int tile_size         = threads_per_block * items_per_thread;

  const int num_valid = GENERATE_COPY(values<int>({threads_per_block, tile_size - items_per_thread, tile_size}));
  const int k =
    GENERATE_COPY(values<int>({1, 5, items_per_thread, threads_per_block, num_valid / 2, num_valid - 1, num_valid}));
  const int overhang = GENERATE_COPY(overhang_generator(k == num_valid, {0, 1, (num_valid - k) / 2, num_valid - k}));
  const bool blocked_input = GENERATE_COPY(true, false);

  // Same shape assertion as test 3: `has_ties == (overhang > 0)`.
  auto run_check = [&](rng_t& rng, key_t boundary_key) {
    CAPTURE(c2h::type_name<key_t>(), select_max, num_valid, k, overhang, blocked_input, boundary_key);
    static constexpr bool is_full_tile = false;
    static constexpr bool blocked      = true;
    static constexpr bool striped      = false;
    c2h::host_vector<key_t> h_in = gen_keys_from_boundary_key<select_max>(num_valid, k, overhang, boundary_key, rng);
    const auto h_ref             = sorted_top_k<select_max>(h_in, k);
    const bool has_ties =
      blocked_input
        ? check_single_key<key_t, threads_per_block, items_per_thread, is_full_tile, blocked, select_max>(
            h_in, to_span(h_ref), k)
        : check_single_key<key_t, threads_per_block, items_per_thread, is_full_tile, striped, select_max>(
            h_in, to_span(h_ref), k);
    REQUIRE(has_ties == (overhang > 0));
  };

  // Split the boundary_key loop, see test 3.
  SECTION("fixed boundary_key")
  {
    rng_t rng(static_cast<cuda::std::uint32_t>(C2H_SEED(1).get()));
    const key_t boundary_key = GENERATE_COPY(boundary_key_generator<key_t>());
    run_check(rng, boundary_key);
  }
  SECTION("random boundary_key")
  {
    rng_t rng(static_cast<cuda::std::uint32_t>(C2H_SEED(2).get()));
    const key_t boundary_key = random_boundary_key<key_t>(rng);
    run_check(rng, boundary_key);
  }
}

// ---------------------------------------------------------------------------
// 5) Multi-key tie-break: many primary ties, unique secondary breaks them.
//    Sweeps `select_max`/`select_min` and full / partial tiles.
// ---------------------------------------------------------------------------

C2H_TEST("block_topk_sieve resolves primary ties via a secondary refine",
         "[block][topk][rank][multi-key]",
         select_direction_max)
{
  using primary_t                  = float;
  using secondary_t                = cuda::std::uint32_t;
  static constexpr bool select_max = c2h::get<0, TestType>::value;
  // Odd items-per-thread (3) to verify nothing assumes power-of-two strides.
  static constexpr int threads_per_block = 128;
  static constexpr int items_per_thread  = 3;
  static constexpr int tile_size         = threads_per_block * items_per_thread;

  // `num_valid == tile_size` exercises the `IsFullTile=true` instantiation;
  // smaller values exercise `IsFullTile=false`.
  const int num_valid =
    GENERATE_COPY(values<int>({threads_per_block - 1, threads_per_block, tile_size - items_per_thread, tile_size}));

  // Primary: bit-equal boundary copies + strict above/below fillers ->
  // ties iff `overhang > 0`. Secondary: shuffled `[0, num_valid)` ->
  // refine collapses every tied primary class to a unique representative,
  // so post-refine `has_ties()` is always `false`.
  const int k        = GENERATE_COPY(values<int>({1, 7, num_valid / 4, num_valid - 1}));
  const int overhang = GENERATE_COPY(overhang_generator(k == num_valid, {0, 1, (num_valid - k) / 2}));

  auto run_check = [&](rng_t& rng, primary_t boundary_key) {
    const bool is_full_tile = (num_valid == tile_size);
    CAPTURE(select_max, is_full_tile, num_valid, k, overhang, boundary_key);

    c2h::host_vector<primary_t> h_primary =
      gen_keys_from_boundary_key<select_max>(num_valid, k, overhang, boundary_key, rng);
    c2h::host_vector<secondary_t> h_secondary = distinct_keys<secondary_t>(num_valid, rng);

    c2h::device_vector<primary_t> d_primary(h_primary);
    c2h::device_vector<secondary_t> d_secondary(h_secondary);
    c2h::device_vector<primary_t> d_top_primary(k, primary_t{});
    c2h::device_vector<secondary_t> d_top_secondary(k, secondary_t{});
    c2h::device_vector<bool> d_has_ties(2, false);

    // Generic lambda dispatches on the runtime `is_full_tile` flag to one of
    // the two compile-time `IsFullTile` kernel instantiations.
    auto launch_kernel = [&](auto is_full_tile_const) {
      static constexpr bool ift = decltype(is_full_tile_const)::value;
      multi_key_kernel<primary_t, secondary_t, threads_per_block, items_per_thread, ift, select_max>
        <<<1, threads_per_block>>>(
          to_span(d_primary),
          to_span(d_secondary),
          to_span(d_top_primary),
          to_span(d_top_secondary),
          cuda::std::span<bool, 2>{cuda::std::to_address(d_has_ties.data()), 2});
    };
    if (is_full_tile)
    {
      launch_kernel(cuda::std::true_type{});
    }
    else
    {
      launch_kernel(cuda::std::false_type{});
    }
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    const c2h::host_vector<bool> h_has_ties(d_has_ties);
    CAPTURE(h_has_ties[0], h_has_ties[1]);
    REQUIRE(h_has_ties[0] == (overhang > 0));
    REQUIRE_FALSE(h_has_ties[1]);

    auto zip_ref = thrust::make_zip_iterator(h_primary.begin(), h_secondary.begin());
    thrust::sort(zip_ref, zip_ref + num_valid, comparator_t<select_max>{});
    h_primary.resize(k);
    h_secondary.resize(k);

    c2h::host_vector<primary_t> h_top_primary(d_top_primary);
    c2h::host_vector<secondary_t> h_top_secondary(d_top_secondary);
    auto zip_got = thrust::make_zip_iterator(h_top_primary.begin(), h_top_secondary.begin());
    thrust::sort(zip_got, zip_got + k, comparator_t<select_max>{});

    REQUIRE(h_primary == h_top_primary);
    REQUIRE(h_secondary == h_top_secondary);
  };

  // Split the boundary_key loop, see test 3.
  SECTION("fixed boundary_key")
  {
    rng_t rng(static_cast<cuda::std::uint32_t>(C2H_SEED(1).get()));
    const primary_t boundary_key = GENERATE_COPY(boundary_key_generator<primary_t>());
    run_check(rng, boundary_key);
  }
  SECTION("random boundary_key")
  {
    rng_t rng(static_cast<cuda::std::uint32_t>(C2H_SEED(2).get()));
    const primary_t boundary_key = random_boundary_key<primary_t>(rng);
    run_check(rng, boundary_key);
  }
}

// ---------------------------------------------------------------------------
// 6) Bit-window split: chained `refine_max`'es over disjoint windows must
//    match a one-shot `select_max`. SECTIONs cover the two ends of
//    `bit_window_kernel`'s loop, pinned via `g_last_hi`.
// ---------------------------------------------------------------------------

C2H_TEST("block_topk_sieve produces the same selection across bit-window splits", "[block][topk][rank][bit-window]")
{
  using key_t                            = cuda::std::uint32_t;
  static constexpr int threads_per_block = 64;
  static constexpr int items_per_thread  = 6;
  static constexpr int tile_size         = threads_per_block * items_per_thread;
  static constexpr int window_bits       = 8;
  static constexpr int key_bits          = static_cast<int>(sizeof(key_t) * 8);
  static constexpr bool is_full_tile     = true;
  static constexpr bool blocked_input    = true;
  static constexpr bool select_max       = true;

  // Compares `bit_window_kernel` vs. one-shot `single_key_kernel`, pinning
  // post-loop `hi` to `expected_last_hi` (`0` = full chain ran).
  auto run_compare = [&](const c2h::host_vector<key_t>& h_in, int k, int expected_last_hi) {
    c2h::device_vector<key_t> d_in(h_in);
    c2h::device_vector<key_t> d_top_split(k, key_t{});
    c2h::device_vector<key_t> d_top_oneshot(k, key_t{});
    // `single_key_kernel`'s round-trip output is unused; scratch.
    c2h::device_vector<key_t> d_keys_out(tile_size, key_t{});
    c2h::device_vector<bool> d_has_ties(1, false);
    c2h::device_vector<int> d_last_hi(1, -1);

    // One-shot reference.
    single_key_kernel<key_t, threads_per_block, items_per_thread, is_full_tile, blocked_input, select_max>
      <<<1, threads_per_block>>>(
        to_span(d_in),
        to_span(d_keys_out),
        to_span(d_top_oneshot),
        cuda::std::span<bool, 1>{cuda::std::to_address(d_has_ties.data()), 1});
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    bit_window_kernel<key_t, threads_per_block, items_per_thread, window_bits><<<1, threads_per_block>>>(
      to_span(d_in), to_span(d_top_split), cuda::std::span<int, 1>{cuda::std::to_address(d_last_hi.data()), 1});
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    const int last_hi = c2h::host_vector<int>(d_last_hi)[0];
    CAPTURE(last_hi, expected_last_hi);
    REQUIRE(last_hi == expected_last_hi);

    c2h::host_vector<key_t> h_top_oneshot(d_top_oneshot);
    c2h::host_vector<key_t> h_top_split(d_top_split);
    std::sort(h_top_oneshot.begin(), h_top_oneshot.end());
    std::sort(h_top_split.begin(), h_top_split.end());
    REQUIRE(h_top_oneshot == h_top_split);
  };

  SECTION("ties to the last window")
  {
    const int k = GENERATE_COPY(values<int>({1, 11, tile_size / 4, tile_size - 1}));
    // `overhang == 0` excluded: would let the early exit fire non-
    // deterministically. `> 0` keeps `has_ties()` true to `hi == 0`.
    const int overhang = GENERATE_COPY(overhang_generator(k == tile_size, {1, (tile_size - k) / 2}));
    // `key_t` is unsigned -- only the random draw applies.
    rng_t rng(static_cast<cuda::std::uint32_t>(C2H_SEED(1).get()));
    const key_t boundary_key = random_boundary_key<key_t>(rng);
    CAPTURE(k, overhang, boundary_key);

    c2h::host_vector<key_t> h_in   = gen_keys_from_boundary_key<select_max>(tile_size, k, overhang, boundary_key, rng);
    constexpr int expected_last_hi = 0;
    run_compare(h_in, k, expected_last_hi);
  }

  SECTION("early exit before last window")
  {
    // Distinct values `i in [0, tile_size)` shifted left by `2 * window_bits`,
    // confining entropy to bits [16, 25). `select_max` over [24, 32) at most
    // partitions candidates on bit 24; within each partition, `[16, 24)`
    // bits are still distinct, so the iter-1 refine over [16, 24) fully
    // orders the top-k. The iter-2 `!has_ties()` break then fires before
    // visiting windows [8, 16) and [0, 8).
    const int k = GENERATE_COPY(values<int>({1, 11, tile_size / 4, tile_size - 1}));
    rng_t rng(static_cast<cuda::std::uint32_t>(C2H_SEED(1).get()));
    CAPTURE(k);

    c2h::host_vector<key_t> h_in(tile_size);
    for (int i = 0; i < tile_size; ++i)
    {
      h_in[i] = static_cast<key_t>(i << (2 * window_bits));
    }
    thrust::shuffle(h_in.begin(), h_in.end(), rng);
    constexpr int expected_last_hi = key_bits - 2 * window_bits;
    run_compare(h_in, k, expected_last_hi);
  }
}
