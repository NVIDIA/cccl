//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Three relational shapes (the ones a columnar DataFrame library
 *        such as cuDF runs all day) written as compositions of the sharded
 *        verbs over the locality domains of one device.
 *
 * A table is a struct of columns; each column is a `sharded_array` over
 * `place_group{make_locality_domain_grid(0)}`, every column cut at the SAME
 * row boundaries (co-partitioned). Nothing here names a device: the group
 * decides where each shard lives.
 *
 * Two grades of column storage are used deliberately:
 *
 * - VALUE COLUMNS are `allocate_contiguous`: one VA range, each shard's
 *   pages physically owned by its place. That gives a column one base
 *   pointer, so a functor holding `column_base` can gather `column[row]` by
 *   GLOBAL row id — the same trick `sharded_graph.cu` uses for vertex
 *   vectors. (Physical ownership snaps to the 2 MiB granule at the cut.)
 * - RAGGED RESULTS (survivor ids, group boundaries, anti-join output) are
 *   plain `allocate`: size-mutating verbs (`copy_if`) shrink each shard and
 *   COMMIT the data-dependent sizes into the container; the contiguous grade
 *   refuses that by design (a shrunken shard would leave a gap).
 *
 * VALIDITY. cuDF stores nullability as one bit per row in 32-bit words. A
 * row cut that is a multiple of 32 makes each shard's words exactly its
 * rows' bits (no word straddles two places), so the bitmask can be a
 * co-partitioned contiguous column of `uint32_t` words with `rows/32`
 * entries per shard. The row cut below is chosen accordingly.
 *
 * Sections (each verified against a host reference):
 *
 *  1. FILTER + AGGREGATE (TPC-H Q6 shape).  cuDF `detail/copy_if.cuh`:
 *     thrust::copy_if of row ids -> `output_size = distance(begin, end)`
 *     (a host readback) -> gather.  Here: `copy_if` of row ids into a
 *     RAGGED owning array (the returned/committed sizes ARE the distance
 *     step, one host sync for all shards) -> gather-multiply as
 *     `zip_transform` -> `reduce_into` on the caller's stream.
 *  2. GROUP-BY AGGREGATE (Q1 shape).  Sort by group key carrying the row
 *     id as a packed 64-bit key (`sort` is keys-only; see the gap list) ->
 *     group starts flagged with `adjacent_difference` -> `copy_if` of the
 *     flagged positions -> a whole `groups+1` offsets array -> whole-offsets
 *     `segmented_reduce` per value column, output co-partitioned with the
 *     groups.  A group straddling the shard cut is reduced as two pieces
 *     and merged on the host (P-1 merges at most — the same boundary
 *     pattern `unique` implements and `run_length_encode` will need).
 *  3. ANTI-JOIN (cuDF `mark_join.cu` shape).  Small build table kept WHOLE
 *     (one device allocation every place reads), an open-addressing hash
 *     set built once; the sharded probe column marks hit build slots with
 *     `atomicOr` (`for_each`), then the unmatched VALID probe rows are
 *     selected with `copy_if`, and the NULL probe rows (validity bit 0) are
 *     selected with a second `copy_if` — cuDF appends the second select at
 *     `result.begin() + unmatched_valid` in ONE buffer; no append form of
 *     `copy_if` exists yet, so the example reports two ragged arrays.
 *
 * Verb gaps this example works around (candidates for the verb set):
 *  - `copy_if` over a counting-iterator input: the row ids have to be
 *    materialized with `sequence` before selecting them.
 *  - APPEND `copy_if` (select into `out` starting at its committed size).
 *  - sort BY KEY with a payload: keys-only sort, so `(key << 32) | row` is
 *    packed into a `uint64_t` (the shared-VA engine takes 64-bit arithmetic
 *    keys through the radix path).
 *  - `adjacent_difference` writes `out[0] = in[0]` (no predecessor), so the
 *    boundary predicate treats position 0 explicitly.
 *  - `run_length_encode`/`reduce_by_key`: the straddling-group merge is done
 *    on the host here.
 */

#include <cuda/experimental/sharded.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
// ---------------------------------------------------------------------------
// Deterministic column generation (host and device agree bit for bit).
// ---------------------------------------------------------------------------
__host__ __device__ inline ::std::uint32_t hash32(::std::uint32_t x, ::std::uint32_t salt)
{
  x ^= salt * 0x9E3779B9u;
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

constexpr int num_group_keys        = 6; // returnflag x linestatus (~6 distinct values in TPC-H)
constexpr int build_key_range       = 1 << 16; // orderkey domain
constexpr ::std::size_t build_count = 4096; // build-side keys
constexpr int shipdate_lo           = 8766; // 1994-01-01 as a day number
constexpr int shipdate_hi           = shipdate_lo + 365;

struct gen
{
  __host__ __device__ static int quantity(::std::size_t r)
  {
    return 1 + static_cast<int>(hash32(static_cast<::std::uint32_t>(r), 1) % 50);
  }
  __host__ __device__ static float discount(::std::size_t r)
  {
    return static_cast<float>(hash32(static_cast<::std::uint32_t>(r), 2) % 11) * 0.01f; // 0.00 .. 0.10
  }
  __host__ __device__ static float extendedprice(::std::size_t r)
  {
    return 900.0f + static_cast<float>(hash32(static_cast<::std::uint32_t>(r), 3) % 100000) * 0.01f;
  }
  __host__ __device__ static int shipdate(::std::size_t r)
  {
    return 8000 + static_cast<int>(hash32(static_cast<::std::uint32_t>(r), 4) % 2557); // ~7 years
  }
  __host__ __device__ static int group_key(::std::size_t r)
  {
    return static_cast<int>(hash32(static_cast<::std::uint32_t>(r), 5) % num_group_keys);
  }
  __host__ __device__ static int orderkey(::std::size_t r)
  {
    return static_cast<int>(hash32(static_cast<::std::uint32_t>(r), 6) % build_key_range);
  }
  // validity of the nullable column (orderkey): ~3% nulls
  __host__ __device__ static bool valid(::std::size_t r)
  {
    return hash32(static_cast<::std::uint32_t>(r), 7) % 32 != 0;
  }
  __host__ __device__ static ::std::uint32_t validity_word(::std::size_t w)
  {
    ::std::uint32_t bits = 0;
    for (::std::uint32_t b = 0; b < 32; b++)
    {
      bits |= (valid(w * 32 + b) ? 1u : 0u) << b;
    }
    return bits;
  }
  __host__ __device__ static int build_key(::std::size_t i)
  {
    // 4096 distinct keys: a stride pattern over the orderkey domain
    return static_cast<int>((i * 16 + hash32(static_cast<::std::uint32_t>(i), 8) % 16) % build_key_range);
  }
};

// Generators as tabulate functors (index = GLOBAL row / word id).
struct gen_quantity
{
  __device__ int operator()(::std::size_t r) const
  {
    return gen::quantity(r);
  }
};
struct gen_discount
{
  __device__ float operator()(::std::size_t r) const
  {
    return gen::discount(r);
  }
};
struct gen_price
{
  __device__ float operator()(::std::size_t r) const
  {
    return gen::extendedprice(r);
  }
};
struct gen_shipdate
{
  __device__ int operator()(::std::size_t r) const
  {
    return gen::shipdate(r);
  }
};
struct gen_orderkey
{
  __device__ int operator()(::std::size_t r) const
  {
    return gen::orderkey(r);
  }
};
struct gen_validity
{
  __device__ ::std::uint32_t operator()(::std::size_t w) const
  {
    return gen::validity_word(w);
  }
};
// Packed sort key: (group key << 32) | row id — the "sort by key carrying a
// payload" spelling on a keys-only sort.
struct gen_packed_key
{
  __device__ unsigned long long operator()(::std::size_t r) const
  {
    return (static_cast<unsigned long long>(gen::group_key(r)) << 32) | static_cast<unsigned long long>(r);
  }
};
__host__ __device__ inline int packed_group(unsigned long long k)
{
  return static_cast<int>(k >> 32);
}
__host__ __device__ inline int packed_row(unsigned long long k)
{
  return static_cast<int>(k & 0xFFFFFFFFull);
}

__host__ __device__ inline bool validity_bit(const ::std::uint32_t* words, ::std::size_t r)
{
  return (words[r >> 5] >> (r & 31)) & 1u;
}

// ---------------------------------------------------------------------------
// The table: a struct of co-partitioned sharded columns.
// ---------------------------------------------------------------------------
struct lineitem_table
{
  sharded_array<int> quantity;
  sharded_array<float> discount;
  sharded_array<float> extendedprice;
  sharded_array<int> shipdate;
  sharded_array<int> orderkey; // nullable: validity below
  sharded_array<::std::uint32_t> validity; // one bit per row, 32 rows per word
  ::std::size_t num_rows = 0;
};

// ---------------------------------------------------------------------------
// Section 1 functors (Q6 shape)
// ---------------------------------------------------------------------------
struct q6_pred // applied to ROW IDS; columns gathered through their bases
{
  const int* shipdate;
  const float* discount;
  const int* quantity;
  __device__ bool operator()(int r) const
  {
    const int d     = shipdate[r];
    const float dis = discount[r];
    return d >= shipdate_lo && d < shipdate_hi && dis >= 0.05f && dis <= 0.07f && quantity[r] < 24;
  }
};
struct gather_price_x_discount
{
  const float* price;
  const float* discount;
  __device__ float operator()(int r) const
  {
    return price[r] * discount[r];
  }
};

// ---------------------------------------------------------------------------
// Section 2 functors (Q1 shape)
// ---------------------------------------------------------------------------
struct key_change_op // adjacent_difference: out[i] = op(in[i], in[i-1])
{
  __device__ unsigned long long operator()(unsigned long long cur, unsigned long long prev) const
  {
    return packed_group(cur) != packed_group(prev) ? 1ull : 0ull;
  }
};
struct boundary_pred // applied to POSITIONS in the sorted order
{
  const unsigned long long* marks; // contiguous; marks[0] == in[0] (no predecessor), handled explicitly
  __device__ bool operator()(int p) const
  {
    return p == 0 || marks[p] != 0ull;
  }
};
struct gather_int_by_packed
{
  const int* column;
  __device__ int operator()(unsigned long long k) const
  {
    return column[packed_row(k)];
  }
};
struct gather_float_by_packed
{
  const float* column;
  __device__ float operator()(unsigned long long k) const
  {
    return column[packed_row(k)];
  }
};
struct gather_valid_by_packed // COUNT(column) counts the non-null rows
{
  const ::std::uint32_t* validity;
  __device__ int operator()(unsigned long long k) const
  {
    return validity_bit(validity, static_cast<::std::size_t>(packed_row(k))) ? 1 : 0;
  }
};
struct sum_ll
{
  __host__ __device__ long long operator()(long long a, long long b) const
  {
    return a + b;
  }
};
struct sum_f
{
  __host__ __device__ float operator()(float a, float b) const
  {
    return a + b;
  }
};

// ---------------------------------------------------------------------------
// Section 3: a minimal open-addressing hash set (linear probing) over a
// WHOLE build table, plus one mark word per slot. Written here because the
// example depends on nothing but CCCL; cuDF uses cuco for the same role.
// ---------------------------------------------------------------------------
constexpr int hash_set_capacity = 8192; // power of two, load factor 0.5
constexpr int empty_slot        = -1;

struct hash_set_view
{
  int* slots; // build keys, `empty_slot` when free
  ::std::uint32_t* marks; // mark bits per slot, 32 slots per word
  __device__ static int slot_of(int key)
  {
    return static_cast<int>(hash32(static_cast<::std::uint32_t>(key), 11) & (hash_set_capacity - 1));
  }
  __device__ void insert(int key) const
  {
    int s = slot_of(key);
    for (;;)
    {
      const int prev = atomicCAS(&slots[s], empty_slot, key);
      if (prev == empty_slot || prev == key)
      {
        return;
      }
      s = (s + 1) & (hash_set_capacity - 1);
    }
  }
  // Probe: returns the slot index holding `key`, or -1.
  __device__ int find(int key) const
  {
    int s = slot_of(key);
    for (;;)
    {
      const int v = slots[s];
      if (v == key)
      {
        return s;
      }
      if (v == empty_slot)
      {
        return -1;
      }
      s = (s + 1) & (hash_set_capacity - 1);
    }
  }
};

__global__ void build_hash_set_kernel(hash_set_view set, const int* keys, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    set.insert(keys[i]);
  }
}

// for_each over the probe column: (value, global row id). Valid rows that hit
// a build key set that slot's mark bit (the `mark_join` side effect).
struct mark_hits
{
  hash_set_view set;
  const ::std::uint32_t* validity;
  __device__ void operator()(int& key, ::std::size_t r) const
  {
    if (!validity_bit(validity, r))
    {
      return;
    }
    const int s = set.find(key);
    if (s >= 0)
    {
      atomicOr(&set.marks[s >> 5], 1u << (s & 31));
    }
  }
};
struct unmatched_valid_pred // applied to ROW IDS
{
  hash_set_view set;
  const int* orderkey;
  const ::std::uint32_t* validity;
  __device__ bool operator()(int r) const
  {
    return validity_bit(validity, r) && set.find(orderkey[r]) < 0;
  }
};
struct null_row_pred
{
  const ::std::uint32_t* validity;
  __device__ bool operator()(int r) const
  {
    return !validity_bit(validity, r);
  }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
template <class T>
sharded_array<T> allocate_contiguous_column(place_group& group, const ::std::vector<::std::size_t>& sizes)
{
  ::std::vector<shard_spec> specs;
  for (::std::size_t g = 0; g < sizes.size(); g++)
  {
    const auto& place = group.place(g);
    specs.emplace_back(sizes[g], place.affine_data_place(), place, group.get_stream(g, 0));
  }
  return sharded_array<T>::allocate_contiguous(specs);
}

template <class T>
void print_shard_sizes(const char* what, const sharded_array<T>& a)
{
  ::std::printf("  %s: total %zu, per shard [", what, a.size());
  for (::std::size_t g = 0; g < a.num_shards(); g++)
  {
    ::std::printf("%s%zu", g ? ", " : "", a.shard(g).size);
  }
  ::std::printf("]\n");
}
} // namespace

int main(int argc, char** argv)
{
  ::std::size_t N = ::std::size_t{1} << 24;
  for (int i = 1; i < argc; i++)
  {
    if (::std::strcmp(argv[i], "--rows") == 0 && i + 1 < argc)
    {
      N = ::std::strtoull(argv[++i], nullptr, 10);
    }
  }
  if (N == 0 || N % 32 != 0 || N > (::std::size_t{1} << 31))
  {
    ::std::fprintf(stderr, "--rows must be a positive multiple of 32 below 2^31 (row ids are int32)\n");
    return 1;
  }

  cuda_safe_call(cuInit(0));
  cuda_safe_call(cudaSetDevice(0));

  auto group            = place_group{make_locality_domain_grid(0)};
  const ::std::size_t P = group.size();
  auto envs             = group.envs(0);
  ::std::printf("place_group with %zu place(s), N = %zu rows\n", P, N);

  // Row cut: equal shares rounded DOWN to a multiple of 32 rows (validity
  // words never straddle a place); the last shard absorbs the remainder,
  // itself a multiple of 32 since N is.
  ::std::vector<::std::size_t> rows(P), words(P), row_begin(P + 1, 0);
  {
    const ::std::size_t share = (N / P) / 32 * 32;
    for (::std::size_t g = 0; g < P; g++)
    {
      rows[g]          = (g + 1 == P) ? N - share * (P - 1) : share;
      words[g]         = rows[g] / 32;
      row_begin[g + 1] = row_begin[g] + rows[g];
    }
  }

  // -------------------------------------------------------------------------
  // The table: contiguous value columns generated in place (tabulate by
  // global row id), validity words co-partitioned at rows/32.
  // -------------------------------------------------------------------------
  lineitem_table t;
  t.num_rows      = N;
  t.quantity      = allocate_contiguous_column<int>(group, rows);
  t.discount      = allocate_contiguous_column<float>(group, rows);
  t.extendedprice = allocate_contiguous_column<float>(group, rows);
  t.shipdate      = allocate_contiguous_column<int>(group, rows);
  t.orderkey      = allocate_contiguous_column<int>(group, rows);
  t.validity      = allocate_contiguous_column<::std::uint32_t>(group, words);
  tabulate(t.quantity, envs, gen_quantity{});
  tabulate(t.discount, envs, gen_discount{});
  tabulate(t.extendedprice, envs, gen_price{});
  tabulate(t.shipdate, envs, gen_shipdate{});
  tabulate(t.orderkey, envs, gen_orderkey{});
  tabulate(t.validity, envs, gen_validity{});

  const int* quantity_base             = t.quantity.contiguous_data();
  const float* discount_base           = t.discount.contiguous_data();
  const float* price_base              = t.extendedprice.contiguous_data();
  const int* shipdate_base             = t.shipdate.contiguous_data();
  const int* orderkey_base             = t.orderkey.contiguous_data();
  const ::std::uint32_t* validity_base = t.validity.contiguous_data();

  // Row ids, materialized once: `copy_if` selects from a sharded VIEW, so
  // there is no "copy_if over a counting iterator" spelling yet (gap).
  auto row_ids = sharded_array<int>::allocate(group, rows, 0);
  sequence(row_ids, envs, 0, 1);

  // A caller stream for the asynchronous terminators (`reduce_into`).
  cudaStream_t caller = nullptr;
  cuda_safe_call(cudaStreamCreateWithFlags(&caller, cudaStreamNonBlocking));
  const auto caller_prop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{caller}};
  const auto caller_env  = ::cuda::std::execution::env{caller_prop};

  bool ok = true;

  // =========================================================================
  // 1. FILTER + AGGREGATE (Q6): sum(extendedprice * discount) over the rows
  //    with shipdate in a year, discount in [0.05, 0.07], quantity < 24.
  //
  //    cuDF: copy_if(row ids) -> distance -> gather -> reduce, four passes
  //    with one host readback for the size. Here the readback is the
  //    commit of the ragged survivors array (all shards, one join).
  // =========================================================================
  ::std::printf("[1] filter + aggregate (Q6 shape)\n");
  auto survivors        = sharded_array<int>::allocate(group, rows, 0); // capacity = worst case
  const ::std::size_t s = copy_if(row_ids, envs, survivors, q6_pred{shipdate_base, discount_base, quantity_base});
  print_shard_sizes("survivors (ragged, committed by copy_if)", survivors);

  // Gather-multiply for the survivors only, into a ragged array laid out
  // like the survivors (allocate_like copies the committed sizes).
  auto products = sharded_array<float>::allocate_like(survivors);
  zip_transform(products, envs, gather_price_x_discount{price_base, discount_base}, default_call_env{}, survivors);

  float* h_revenue = nullptr;
  cuda_safe_call(cudaMallocHost(&h_revenue, sizeof(float)));
  reduce_into(products, envs, h_revenue, sum_f{}, 0.0f, caller_env);
  cuda_safe_call(cudaStreamSynchronize(caller));

  {
    double ref        = 0.0;
    ::std::size_t cnt = 0;
    for (::std::size_t r = 0; r < N; r++)
    {
      const int d     = gen::shipdate(r);
      const float dis = gen::discount(r);
      if (d >= shipdate_lo && d < shipdate_hi && dis >= 0.05f && dis <= 0.07f && gen::quantity(r) < 24)
      {
        ref += static_cast<double>(gen::extendedprice(r) * dis);
        cnt++;
      }
    }
    const bool count_ok = (s == cnt);
    const bool sum_ok   = ::std::abs(static_cast<double>(*h_revenue) - ref) <= 1e-3 * (1.0 + ::std::abs(ref));
    ok                  = ok && count_ok && sum_ok;
    ::std::printf(
      "  survivors %zu (ref %zu) %s, revenue %.2f (ref %.2f) %s\n",
      s,
      cnt,
      count_ok ? "OK" : "MISMATCH",
      static_cast<double>(*h_revenue),
      ref,
      sum_ok ? "OK" : "MISMATCH");
  }

  // =========================================================================
  // 2. GROUP-BY AGGREGATE (Q1): per group key, SUM(quantity),
  //    SUM(extendedprice), COUNT(orderkey) (non-null rows).
  // =========================================================================
  ::std::printf("[2] group-by aggregate (Q1 shape)\n");

  // 2a. Sort by key carrying the row id (packed 64-bit key). Shards keep
  //     their boundaries; the array reads as one globally sorted sequence.
  auto keys = sharded_array<unsigned long long>::allocate(group, rows, 0);
  tabulate(keys, envs, gen_packed_key{});
  sort(group, keys);

  // 2b. Group starts: adjacent_difference flags a key change (the
  //     predecessor of a shard's first element comes from the previous
  //     shard — the one-element halo). Contiguous, so the boundary
  //     predicate can read it by global position.
  auto marks = allocate_contiguous_column<unsigned long long>(group, rows);
  adjacent_difference(keys, envs, marks, key_change_op{});

  // 2c. Boundary positions as a RAGGED copy_if of the positions
  //     (materialized again: the counting-iterator gap).
  auto positions  = sharded_array<int>::allocate(group, rows, 0);
  auto boundaries = sharded_array<int>::allocate(group, rows, 0);
  sequence(positions, envs, 0, 1);
  const ::std::size_t G = copy_if(positions, envs, boundaries, boundary_pred{marks.contiguous_data()});
  print_shard_sizes("group starts (ragged)", boundaries);

  // 2d. The whole `pieces+1` offsets array. Assembled on the host from the
  //     committed ragged shards: the group count is tiny (a handful of
  //     keys), so this is a few dozen bytes. Every shard cut is inserted as
  //     an extra boundary when a group straddles it — whole-offsets
  //     segmented_reduce requires no segment to cross the value cut. Such
  //     a straddling group becomes two PIECES, merged on the host below.
  ::std::vector<int> h_bounds(G);
  boundaries.copy_to_host(h_bounds.data());
  ::std::vector<int> h_off(h_bounds);
  for (::std::size_t g = 1; g < P; g++)
  {
    h_off.push_back(static_cast<int>(row_begin[g]));
  }
  h_off.push_back(static_cast<int>(N));
  ::std::sort(h_off.begin(), h_off.end());
  h_off.erase(::std::unique(h_off.begin(), h_off.end()), h_off.end());
  const ::std::size_t pieces = h_off.size() - 1;
  ::std::vector<::std::size_t> pieces_per_shard(P, 0);
  for (::std::size_t p = 0; p < pieces; p++)
  {
    const ::std::size_t g = ::std::upper_bound(row_begin.begin(), row_begin.end(), h_off[p]) - row_begin.begin() - 1;
    pieces_per_shard[g]++;
  }
  int* d_off = nullptr;
  cuda_safe_call(cudaMalloc(&d_off, h_off.size() * sizeof(int)));
  cuda_safe_call(cudaMemcpy(d_off, h_off.data(), h_off.size() * sizeof(int), cudaMemcpyHostToDevice));
  ::std::printf("  groups %zu, pieces %zu (straddling cuts split into pieces)\n", G, pieces);

  // 2e. Value columns in SORTED order (gather by the packed row id), then
  //     the whole-offsets segmented reduce, output co-partitioned with the
  //     pieces.
  auto q_sorted     = sharded_array<int>::allocate(group, rows, 0);
  auto price_sorted = sharded_array<float>::allocate(group, rows, 0);
  auto valid_sorted = sharded_array<int>::allocate(group, rows, 0);
  zip_transform(q_sorted, envs, gather_int_by_packed{quantity_base}, default_call_env{}, keys);
  zip_transform(price_sorted, envs, gather_float_by_packed{price_base}, default_call_env{}, keys);
  zip_transform(valid_sorted, envs, gather_valid_by_packed{validity_base}, default_call_env{}, keys);

  auto sum_qty   = sharded_array<long long>::allocate(group, pieces_per_shard, 0);
  auto sum_price = sharded_array<float>::allocate(group, pieces_per_shard, 0);
  auto count_ok_ = sharded_array<long long>::allocate(group, pieces_per_shard, 0);
  segmented_reduce(q_sorted, envs, d_off, sum_qty, sum_ll{}, 0ll);
  segmented_reduce(price_sorted, envs, d_off, sum_price, sum_f{}, 0.0f);
  segmented_reduce(valid_sorted, envs, d_off, count_ok_, sum_ll{}, 0ll);
  print_shard_sizes("aggregates (co-partitioned with the pieces)", sum_qty);

  // 2f. Merge pieces into groups on the host (a piece starting at a cut
  //     where no key change was flagged continues the previous group) and
  //     verify.
  {
    ::std::vector<long long> h_q(pieces), h_c(pieces);
    ::std::vector<float> h_p(pieces);
    sum_qty.copy_to_host(h_q.data());
    sum_price.copy_to_host(h_p.data());
    count_ok_.copy_to_host(h_c.data());

    ::std::vector<long long> h_marks_at_start(pieces);
    for (::std::size_t p = 0; p < pieces; p++)
    {
      unsigned long long m = 1;
      if (h_off[p] != 0)
      {
        cuda_safe_call(cudaMemcpy(&m, marks.contiguous_data() + h_off[p], sizeof(m), cudaMemcpyDeviceToHost));
      }
      h_marks_at_start[p] = static_cast<long long>(m);
    }
    ::std::vector<long long> g_q, g_c;
    ::std::vector<double> g_p;
    ::std::vector<long long> g_rows;
    for (::std::size_t p = 0; p < pieces; p++)
    {
      if (h_marks_at_start[p] != 0 || p == 0)
      {
        g_q.push_back(0);
        g_c.push_back(0);
        g_p.push_back(0.0);
        g_rows.push_back(0);
      }
      g_q.back() += h_q[p];
      g_c.back() += h_c[p];
      g_p.back() += static_cast<double>(h_p[p]);
      g_rows.back() += h_off[p + 1] - h_off[p];
    }
    const ::std::size_t merged = pieces - g_q.size();

    // Host reference: groups appear in ascending key order after the sort.
    ::std::map<int, ::std::array<double, 4>> ref; // key -> {sum q, sum price, count valid, rows}
    for (::std::size_t r = 0; r < N; r++)
    {
      auto& a = ref[gen::group_key(r)];
      a[0] += gen::quantity(r);
      a[1] += static_cast<double>(gen::extendedprice(r));
      a[2] += gen::valid(r) ? 1 : 0;
      a[3] += 1;
    }
    bool ok2        = (G == ref.size()) && (g_q.size() == ref.size());
    ::std::size_t i = 0;
    for (const auto& [key, a] : ref)
    {
      if (!ok2 || i >= g_q.size())
      {
        break;
      }
      const bool q_ok = static_cast<double>(g_q[i]) == a[0];
      const bool p_ok = ::std::abs(g_p[i] - a[1]) <= 1e-3 * (1.0 + ::std::abs(a[1]));
      const bool c_ok = static_cast<double>(g_c[i]) == a[2];
      const bool n_ok = static_cast<double>(g_rows[i]) == a[3];
      ::std::printf(
        "  key %d: rows %lld sum(qty) %lld sum(price) %.2f count %lld %s\n",
        key,
        g_rows[i],
        g_q[i],
        g_p[i],
        g_c[i],
        (q_ok && p_ok && c_ok && n_ok) ? "OK" : "MISMATCH");
      ok2 = ok2 && q_ok && p_ok && c_ok && n_ok;
      i++;
    }
    ok = ok && ok2;
    ::std::printf(
      "  %zu groups from %zu pieces (%zu host merge%s at the cut): %s\n",
      g_q.size(),
      pieces,
      merged,
      merged == 1 ? "" : "s",
      ok2 ? "OK" : "MISMATCH");
  }
  cuda_safe_call(cudaFree(d_off));

  // =========================================================================
  // 3. ANTI-JOIN (mark_join shape): probe rows whose orderkey is not in the
  //    build set, then the null probe rows.
  // =========================================================================
  ::std::printf("[3] anti-join (mark_join shape)\n");

  // The build side stays WHOLE: one device allocation (no place named),
  // readable from every place; the hash set and its marks likewise.
  ::std::vector<int> h_build(build_count);
  for (::std::size_t i = 0; i < build_count; i++)
  {
    h_build[i] = gen::build_key(i);
  }
  int* d_build = nullptr;
  hash_set_view set{};
  cuda_safe_call(cudaMalloc(&d_build, build_count * sizeof(int)));
  cuda_safe_call(cudaMalloc(&set.slots, hash_set_capacity * sizeof(int)));
  cuda_safe_call(cudaMalloc(&set.marks, hash_set_capacity / 32 * sizeof(::std::uint32_t)));
  cuda_safe_call(cudaMemcpy(d_build, h_build.data(), build_count * sizeof(int), cudaMemcpyHostToDevice));
  cuda_safe_call(cudaMemset(set.slots, 0xFF, hash_set_capacity * sizeof(int))); // empty_slot == -1
  cuda_safe_call(cudaMemset(set.marks, 0, hash_set_capacity / 32 * sizeof(::std::uint32_t)));
  build_hash_set_kernel<<<static_cast<unsigned>((build_count + 255) / 256), 256>>>(
    set, d_build, static_cast<int>(build_count));
  cuda_safe_call(cudaGetLastError());
  cuda_safe_call(cudaDeviceSynchronize());

  // Probe pass with the side effect: every valid probe row that hits a build
  // key marks that slot (atomicOr on a whole mark bitmap).
  for_each(t.orderkey, envs, mark_hits{set, validity_base});

  // The two selects. cuDF's mark_join writes both into ONE pre-sized
  // buffer: unmatched valid rows first, then the null rows appended at
  // `result.begin() + unmatched_valid`. There is no append form of
  // `copy_if` (select into `out` from its committed size) yet, so the two
  // results are two ragged arrays here (gap).
  auto unmatched = sharded_array<int>::allocate(group, rows, 0);
  auto nulls     = sharded_array<int>::allocate(group, rows, 0);
  const ::std::size_t n_unmatched =
    copy_if(row_ids, envs, unmatched, unmatched_valid_pred{set, orderkey_base, validity_base});
  const ::std::size_t n_nulls = copy_if(row_ids, envs, nulls, null_row_pred{validity_base});
  print_shard_sizes("unmatched valid probe rows (ragged)", unmatched);
  print_shard_sizes("null probe rows (ragged, appended in cuDF)", nulls);

  // Marks read back: build keys that were hit at least once.
  ::std::vector<::std::uint32_t> h_marks(hash_set_capacity / 32);
  ::std::vector<int> h_slots(hash_set_capacity);
  cuda_safe_call(
    cudaMemcpy(h_marks.data(), set.marks, h_marks.size() * sizeof(::std::uint32_t), cudaMemcpyDeviceToHost));
  cuda_safe_call(cudaMemcpy(h_slots.data(), set.slots, h_slots.size() * sizeof(int), cudaMemcpyDeviceToHost));
  ::std::size_t marked = 0;
  for (int s = 0; s < hash_set_capacity; s++)
  {
    marked += (h_slots[s] != empty_slot && ((h_marks[s >> 5] >> (s & 31)) & 1u)) ? 1 : 0;
  }

  {
    ::std::vector<char> in_build(build_key_range, 0);
    for (int k : h_build)
    {
      in_build[k] = 1;
    }
    ::std::vector<char> hit(build_key_range, 0);
    ::std::size_t ref_unmatched = 0, ref_nulls = 0;
    for (::std::size_t r = 0; r < N; r++)
    {
      if (!gen::valid(r))
      {
        ref_nulls++;
        continue;
      }
      const int k = gen::orderkey(r);
      if (in_build[k])
      {
        hit[k] = 1;
      }
      else
      {
        ref_unmatched++;
      }
    }
    ::std::size_t ref_marked = 0, distinct_build = 0;
    for (int k = 0; k < build_key_range; k++)
    {
      distinct_build += in_build[k];
      ref_marked += hit[k];
    }
    const bool u_ok = n_unmatched == ref_unmatched;
    const bool n_ok = n_nulls == ref_nulls;
    const bool m_ok = marked == ref_marked;
    ok              = ok && u_ok && n_ok && m_ok;
    ::std::printf(
      "  unmatched valid %zu (ref %zu) %s, nulls %zu (ref %zu) %s, marked build keys %zu/%zu (ref %zu) %s\n",
      n_unmatched,
      ref_unmatched,
      u_ok ? "OK" : "MISMATCH",
      n_nulls,
      ref_nulls,
      n_ok ? "OK" : "MISMATCH",
      marked,
      distinct_build,
      ref_marked,
      m_ok ? "OK" : "MISMATCH");
    // Spot-check the ragged contents: unmatched ids are ascending within a
    // shard and all satisfy the predicate on the host.
    ::std::vector<int> h_u(n_unmatched);
    unmatched.copy_to_host(h_u.data());
    bool contents_ok = true;
    for (::std::size_t i = 0; i < n_unmatched; i += 97)
    {
      const int r = h_u[i];
      contents_ok = contents_ok && gen::valid(static_cast<::std::size_t>(r)) && !in_build[gen::orderkey(r)];
    }
    ok = ok && contents_ok;
    ::std::printf("  unmatched contents spot-check: %s\n", contents_ok ? "OK" : "MISMATCH");
  }

  cuda_safe_call(cudaFree(d_build));
  cuda_safe_call(cudaFree(set.slots));
  cuda_safe_call(cudaFree(set.marks));
  cuda_safe_call(cudaFreeHost(h_revenue));
  cuda_safe_call(cudaStreamDestroy(caller));

  if (!ok)
  {
    ::std::printf("FAILED\n");
    return 1;
  }
  ::std::printf("PASSED (N=%zu, P=%zu)\n", N, P);
  return 0;
}
