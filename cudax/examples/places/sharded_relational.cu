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
 *        verbs over the locality domains of one device — then the same
 *        verbs run over REAL libcudf columns adopted zero-copy.
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
 * A third grade appears in section 4: ADOPTED columns (`sharded_array::adopt`
 * over memory owned by someone else — here libcudf). Adopted and contiguous
 * columns share the property the gathers rely on (one buffer, so the global
 * row id indexes it), which is what `column_base()` abstracts.
 *
 * VALIDITY. cuDF stores nullability as one bit per row in 32-bit words. A
 * row cut that is a multiple of 32 makes each shard's words exactly its
 * rows' bits (no word straddles two places), so the bitmask can be a
 * co-partitioned contiguous column of `uint32_t` words with `rows/32`
 * entries per shard. The row cut below is chosen accordingly.
 *
 * Sections (each verified against a host reference, section 4 against cuDF):
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
 *  4. ADOPT cuDF COLUMNS (compiled only with -DSHARDED_RELATIONAL_WITH_CUDF).
 *     A `cudf::table` with the same lineitem-like columns is built through
 *     libcudf's factories (`make_numeric_column`), the generated data is
 *     copied into its buffers once, and then each `cudf::column_view` is
 *     ADOPTED zero-copy: `column_view` is exactly the shard descriptor by
 *     projection — {data, size, offset, place} — so a `shard<T>` per place
 *     is `{cv.data<T>() + row_begin, rows, row_begin, place}` and
 *     `sharded_array<T>::adopt` wraps them without touching a byte. The
 *     null mask words (`cv.null_mask()`, 32 rows per word) are adopted the
 *     same way. Sections 1 and 2 rerun over the adopted views and are
 *     verified against libcudf's own API (`apply_boolean_mask` +
 *     `binary_operation` + `reduce`; `groupby::aggregate` with SUM/COUNT).
 *
 *     WHICH MEMORY the verbs touch: libcudf allocates through rmm's current
 *     device resource, i.e. whole-device memory INTERLEAVED across the
 *     dies. Adopted cuDF columns are therefore "arm C" of the lab bench:
 *     CONFINEMENT (each place runs only its rows) WITHOUT PLACEMENT (the
 *     rows' pages are spread over both dies). The section then copies the
 *     adopted columns once into placed (`allocate_contiguous`) columns and
 *     reruns Q1 — the "born-placed vs adopted" distinction: same verbs,
 *     same cut, only the page ownership differs. A small timing table
 *     (cudaEvent, median of 5) prints cuDF vs sharded-adopted vs
 *     sharded-placed; informative only, nothing is tuned.
 *
 *     Build (libcudf 26.08, CUDA 13; never add the env's `include/rapids`,
 *     it carries a second CCCL):
 *
 *       nvcc -std=c++20 -O3 -arch=native --expt-relaxed-constexpr --extended-lambda
 *         -DSHARDED_RELATIONAL_WITH_CUDF -DCCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
 *         -D_CCCL_NO_SYSTEM_HEADER -D_CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX
 *         -I<cccl>/cub -I<cccl>/libcudacxx/include -I<cccl>/thrust -isystem <cccl>/cudax/include
 *         -I<cudf-env>/include sharded_relational.cu -o sharded_relational_cudf
 *         -L<cudf-env>/lib -lcudf -lrmm -lcudart -lcuda -Xlinker -rpath=<cudf-env>/lib
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
#include <functional>
#include <map>
#include <stdexcept>
#include <vector>

#ifdef SHARDED_RELATIONAL_WITH_CUDF
#  include <cudf/aggregation.hpp>
#  include <cudf/binaryop.hpp>
#  include <cudf/column/column.hpp>
#  include <cudf/column/column_factories.hpp>
#  include <cudf/column/column_view.hpp>
#  include <cudf/groupby.hpp>
#  include <cudf/null_mask.hpp>
#  include <cudf/reduction.hpp>
#  include <cudf/scalar/scalar.hpp>
#  include <cudf/stream_compaction.hpp>
#  include <cudf/table/table.hpp>
#  include <cudf/table/table_view.hpp>
#  include <cudf/types.hpp>
#  include <cudf/utilities/default_stream.hpp>
#  include <cudf/version_config.hpp>
#  include <rmm/mr/cuda_async_view_memory_resource.hpp>
#  include <rmm/mr/per_device_resource.hpp>
#endif

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
  __host__ __device__ static bool q6_selects(int shipdate, float discount, int quantity)
  {
    return shipdate >= shipdate_lo && shipdate < shipdate_hi && discount >= 0.05f && discount <= 0.07f && quantity < 24;
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
struct gen_group_key
{
  __device__ int operator()(::std::size_t r) const
  {
    return gen::group_key(r);
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
// payload" spelling on a keys-only sort. A zip_transform over the group-key
// column and the row-id column (co-partitioned), so it reads the COLUMN,
// whatever grade of storage backs it.
struct pack_key_op
{
  __device__ unsigned long long operator()(int key, int row) const
  {
    return (static_cast<unsigned long long>(static_cast<unsigned>(key)) << 32)
         | static_cast<unsigned long long>(static_cast<unsigned>(row));
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
// The row cut, shared by every column of a table (co-partitioning).
// ---------------------------------------------------------------------------
struct row_cut
{
  ::std::size_t N = 0;
  ::std::size_t P = 0;
  ::std::vector<::std::size_t> rows, words, row_begin; // row_begin has P+1 entries

  // Equal shares rounded DOWN to a multiple of 32 rows (validity words never
  // straddle a place); the last shard absorbs the remainder, itself a
  // multiple of 32 since N is.
  row_cut(::std::size_t n, ::std::size_t p)
      : N(n)
      , P(p)
      , rows(p)
      , words(p)
      , row_begin(p + 1, 0)
  {
    const ::std::size_t share = (N / P) / 32 * 32;
    for (::std::size_t g = 0; g < P; g++)
    {
      rows[g]          = (g + 1 == P) ? N - share * (P - 1) : share;
      words[g]         = rows[g] / 32;
      row_begin[g + 1] = row_begin[g] + rows[g];
    }
  }
  ::std::size_t shard_of(::std::size_t row) const
  {
    return ::std::upper_bound(row_begin.begin(), row_begin.end(), row) - row_begin.begin() - 1;
  }
};

// ---------------------------------------------------------------------------
// The table: a struct of co-partitioned sharded columns.
// ---------------------------------------------------------------------------
struct lineitem_table
{
  sharded_array<int> quantity;
  sharded_array<float> discount;
  sharded_array<float> extendedprice;
  sharded_array<int> shipdate;
  sharded_array<int> group_key; // returnflag x linestatus, already encoded
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
    return gen::q6_selects(shipdate[r], discount[r], quantity[r]);
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

// The one base pointer a gather-by-global-row functor needs. Two grades
// provide it: the contiguous grade (one VA range, placed pages) and an
// ADOPTED column whose shards are consecutive slices of one foreign buffer
// (section 4). Anything else is refused: a plain `allocate` has one
// allocation per shard, so no global base exists.
template <class T>
const T* column_base(const sharded_array<T>& a)
{
  if (a.is_contiguous())
  {
    return a.contiguous_data();
  }
  const T* base = a.shard(0).data - a.shard(0).global_offset;
  for (::std::size_t g = 0; g < a.num_shards(); g++)
  {
    if (a.shard(g).data != base + a.shard(g).global_offset)
    {
      throw ::std::invalid_argument("column_base: shards are not consecutive slices of one buffer");
    }
  }
  return base;
}

// One element read back by GLOBAL index (whatever the grade): tiny, used for
// the handful of group boundaries.
template <class T>
T read_global(const sharded_array<T>& a, ::std::size_t idx)
{
  for (::std::size_t g = 0; g < a.num_shards(); g++)
  {
    const auto& s = a.shard(g);
    if (s.contains(idx))
    {
      T v{};
      cuda_safe_call(cudaMemcpy(&v, s.data + (idx - s.global_offset), sizeof(T), cudaMemcpyDeviceToHost));
      return v;
    }
  }
  throw ::std::out_of_range("read_global: index outside every shard");
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

// ---------------------------------------------------------------------------
// Q6 as a function of the columns (any grade with a base) — used by section
// 1 over the generated placed columns and by section 4 over adopted cuDF
// columns and their placed copies. `survivors`/`products` are ragged
// workspaces with capacity = rows (the worst case); their sizes are
// re-committed here, so the function can be timed without allocating.
// ---------------------------------------------------------------------------
struct q6_result
{
  ::std::size_t survivors = 0;
  float revenue           = 0.0f;
};

struct q6_workspace
{
  sharded_array<int> survivors;
  sharded_array<float> products;
  float* h_revenue = nullptr; // pinned: `reduce_into` target
};

template <class Envs, class CallEnv>
q6_result run_q6(
  const row_cut& cut,
  const Envs& envs,
  const CallEnv& caller_env,
  cudaStream_t caller,
  const sharded_array<int>& row_ids,
  const sharded_array<int>& shipdate,
  const sharded_array<float>& discount,
  const sharded_array<int>& quantity,
  const sharded_array<float>& price,
  q6_workspace& ws,
  bool verbose)
{
  ws.survivors.commit_sizes(cut.rows); // capacity back to the worst case
  ws.products.commit_sizes(cut.rows);

  // copy_if(row ids) -> distance -> gather -> reduce: four passes in cuDF
  // with one host readback for the size. Here the readback is the commit
  // of the ragged survivors array (all shards, one join).
  q6_result res;
  res.survivors =
    copy_if(row_ids, envs, ws.survivors, q6_pred{column_base(shipdate), column_base(discount), column_base(quantity)});
  if (verbose)
  {
    print_shard_sizes("survivors (ragged, committed by copy_if)", ws.survivors);
  }

  // Gather-multiply for the survivors only, into a ragged array laid out
  // like the survivors (same committed sizes).
  ::std::vector<::std::size_t> sizes(cut.P);
  for (::std::size_t g = 0; g < cut.P; g++)
  {
    sizes[g] = ws.survivors.shard(g).size;
  }
  ws.products.commit_sizes(sizes);
  zip_transform(ws.products,
                envs,
                gather_price_x_discount{column_base(price), column_base(discount)},
                default_call_env{},
                ws.survivors);

  reduce_into(ws.products, envs, ws.h_revenue, sum_f{}, 0.0f, caller_env);
  cuda_safe_call(cudaStreamSynchronize(caller));
  res.revenue = *ws.h_revenue;
  return res;
}

// ---------------------------------------------------------------------------
// Q1 as a function of the columns: groups in ascending key order.
// ---------------------------------------------------------------------------
struct q1_group
{
  int key              = 0;
  long long rows       = 0;
  long long sum_qty    = 0;
  double sum_price     = 0.0;
  long long count      = 0; // COUNT(orderkey): non-null rows
  ::std::size_t pieces = 0; // how many pieces (cut splits) merged into this group
};

struct q1_workspace
{
  sharded_array<unsigned long long> keys;
  sharded_array<unsigned long long> marks; // contiguous: read by global position
  sharded_array<int> boundaries; // ragged
  sharded_array<int> q_sorted, valid_sorted;
  sharded_array<float> price_sorted;
};

template <class Envs>
::std::vector<q1_group> run_q1(
  place_group& group,
  const row_cut& cut,
  const Envs& envs,
  const sharded_array<int>& row_ids,
  const sharded_array<int>& group_key,
  const sharded_array<int>& quantity,
  const sharded_array<float>& price,
  const sharded_array<::std::uint32_t>& validity,
  q1_workspace& ws,
  bool verbose)
{
  const ::std::size_t P = cut.P;

  // a. Sort by key carrying the row id (packed 64-bit key, built from the
  //    group-key COLUMN and the row ids). Shards keep their boundaries; the
  //    array reads as one globally sorted sequence.
  zip_transform(ws.keys, envs, pack_key_op{}, default_call_env{}, group_key, row_ids);
  sort(group, ws.keys);

  // b. Group starts: adjacent_difference flags a key change (the
  //    predecessor of a shard's first element comes from the previous
  //    shard — the one-element halo). Contiguous, so the boundary
  //    predicate can read it by global position.
  adjacent_difference(ws.keys, envs, ws.marks, key_change_op{});

  // c. Boundary positions as a RAGGED copy_if of the positions (the row ids
  //    double as positions: the counting-iterator gap).
  ws.boundaries.commit_sizes(cut.rows);
  const ::std::size_t G = copy_if(row_ids, envs, ws.boundaries, boundary_pred{ws.marks.contiguous_data()});
  if (verbose)
  {
    print_shard_sizes("group starts (ragged)", ws.boundaries);
  }

  // d. The whole `pieces+1` offsets array. Assembled on the host from the
  //    committed ragged shards: the group count is tiny (a handful of
  //    keys), so this is a few dozen bytes. Every shard cut is inserted as
  //    an extra boundary when a group straddles it — whole-offsets
  //    segmented_reduce requires no segment to cross the value cut. Such
  //    a straddling group becomes two PIECES, merged on the host below.
  ::std::vector<int> h_off(G);
  ws.boundaries.copy_to_host(h_off.data());
  for (::std::size_t g = 1; g < P; g++)
  {
    h_off.push_back(static_cast<int>(cut.row_begin[g]));
  }
  h_off.push_back(static_cast<int>(cut.N));
  ::std::sort(h_off.begin(), h_off.end());
  h_off.erase(::std::unique(h_off.begin(), h_off.end()), h_off.end());
  const ::std::size_t pieces = h_off.size() - 1;
  ::std::vector<::std::size_t> pieces_per_shard(P, 0);
  for (::std::size_t p = 0; p < pieces; p++)
  {
    pieces_per_shard[cut.shard_of(static_cast<::std::size_t>(h_off[p]))]++;
  }
  int* d_off = nullptr;
  cuda_safe_call(cudaMalloc(&d_off, h_off.size() * sizeof(int)));
  cuda_safe_call(cudaMemcpy(d_off, h_off.data(), h_off.size() * sizeof(int), cudaMemcpyHostToDevice));
  if (verbose)
  {
    ::std::printf("  groups %zu, pieces %zu (straddling cuts split into pieces)\n", G, pieces);
  }

  // e. Value columns in SORTED order (gather by the packed row id), then the
  //    whole-offsets segmented reduce, output co-partitioned with the pieces.
  zip_transform(ws.q_sorted, envs, gather_int_by_packed{column_base(quantity)}, default_call_env{}, ws.keys);
  zip_transform(ws.price_sorted, envs, gather_float_by_packed{column_base(price)}, default_call_env{}, ws.keys);
  zip_transform(ws.valid_sorted, envs, gather_valid_by_packed{column_base(validity)}, default_call_env{}, ws.keys);

  auto sum_qty   = sharded_array<long long>::allocate(group, pieces_per_shard, 0);
  auto sum_price = sharded_array<float>::allocate(group, pieces_per_shard, 0);
  auto count     = sharded_array<long long>::allocate(group, pieces_per_shard, 0);
  segmented_reduce(ws.q_sorted, envs, d_off, sum_qty, sum_ll{}, 0ll);
  segmented_reduce(ws.price_sorted, envs, d_off, sum_price, sum_f{}, 0.0f);
  segmented_reduce(ws.valid_sorted, envs, d_off, count, sum_ll{}, 0ll);
  if (verbose)
  {
    print_shard_sizes("aggregates (co-partitioned with the pieces)", sum_qty);
  }

  // f. Merge pieces into groups on the host: a piece starting at a cut where
  //    no key change was flagged continues the previous group.
  ::std::vector<long long> h_q(pieces), h_c(pieces);
  ::std::vector<float> h_p(pieces);
  sum_qty.copy_to_host(h_q.data());
  sum_price.copy_to_host(h_p.data());
  count.copy_to_host(h_c.data());
  cuda_safe_call(cudaFree(d_off));

  ::std::vector<q1_group> groups;
  for (::std::size_t p = 0; p < pieces; p++)
  {
    const auto start   = static_cast<::std::size_t>(h_off[p]);
    const bool new_grp = (p == 0) || read_global(ws.marks, start) != 0ull;
    if (new_grp)
    {
      groups.emplace_back();
      groups.back().key = packed_group(read_global(ws.keys, start));
    }
    auto& grp = groups.back();
    grp.sum_qty += h_q[p];
    grp.count += h_c[p];
    grp.sum_price += static_cast<double>(h_p[p]);
    grp.rows += h_off[p + 1] - h_off[p];
    grp.pieces++;
  }
  if (verbose)
  {
    ::std::printf("  %zu groups from %zu pieces (%zu host merge%s at the cut)\n",
                  groups.size(),
                  pieces,
                  pieces - groups.size(),
                  pieces - groups.size() == 1 ? "" : "s");
  }
  return groups;
}

// Q1 verification against a key -> {sum q, sum price, count, rows} reference.
bool check_q1(const ::std::vector<q1_group>& groups,
              const ::std::map<int, ::std::array<double, 4>>& ref,
              const char* ref_name,
              double price_tol)
{
  bool ok         = groups.size() == ref.size();
  ::std::size_t i = 0;
  for (const auto& [key, a] : ref)
  {
    if (!ok || i >= groups.size())
    {
      break;
    }
    const auto& g   = groups[i++];
    const bool k_ok = g.key == key;
    const bool q_ok = static_cast<double>(g.sum_qty) == a[0];
    const bool p_ok = ::std::abs(g.sum_price - a[1]) <= price_tol * (1.0 + ::std::abs(a[1]));
    const bool c_ok = static_cast<double>(g.count) == a[2];
    const bool n_ok = a[3] < 0 || static_cast<double>(g.rows) == a[3];
    ::std::printf(
      "  key %d: rows %lld sum(qty) %lld sum(price) %.2f count %lld  vs %s sum(price) %.2f  %s\n",
      g.key,
      g.rows,
      g.sum_qty,
      g.sum_price,
      g.count,
      ref_name,
      a[1],
      (k_ok && q_ok && p_ok && c_ok && n_ok) ? "OK" : "MISMATCH");
    ok = ok && k_ok && q_ok && p_ok && c_ok && n_ok;
  }
  return ok;
}

q6_workspace make_q6_workspace(place_group& group, const row_cut& cut)
{
  q6_workspace ws;
  ws.survivors = sharded_array<int>::allocate(group, cut.rows, 0); // capacity = worst case
  ws.products  = sharded_array<float>::allocate(group, cut.rows, 0);
  cuda_safe_call(cudaMallocHost(&ws.h_revenue, sizeof(float)));
  return ws;
}

q1_workspace make_q1_workspace(place_group& group, const row_cut& cut)
{
  q1_workspace ws;
  ws.keys         = sharded_array<unsigned long long>::allocate(group, cut.rows, 0);
  ws.marks        = allocate_contiguous_column<unsigned long long>(group, cut.rows);
  ws.boundaries   = sharded_array<int>::allocate(group, cut.rows, 0);
  ws.q_sorted     = sharded_array<int>::allocate(group, cut.rows, 0);
  ws.valid_sorted = sharded_array<int>::allocate(group, cut.rows, 0);
  ws.price_sorted = sharded_array<float>::allocate(group, cut.rows, 0);
  return ws;
}

#ifdef SHARDED_RELATIONAL_WITH_CUDF
// ---------------------------------------------------------------------------
// Section 4 helpers: adopting cuDF columns.
// ---------------------------------------------------------------------------

// `cudf::column_view` -> sharded view, zero-copy. The column's ONE buffer is
// cut at the table's row boundaries; shard g is {cv.data<T>() + row_begin[g],
// rows[g]} executed by place g. The data place is the whole device: cuDF's
// buffer came from rmm's current device resource, whose pages are
// interleaved across the dies — no place owns them (confinement without
// placement). `elements_per_row` = 1 for values, 1/32 for the null mask.
template <class T>
sharded_array<T>
adopt_cudf_buffer(place_group& group, const row_cut& cut, const T* data, const ::std::vector<::std::size_t>& sizes)
{
  ::std::vector<shard<T>> shards(cut.P);
  ::std::size_t begin = 0;
  for (::std::size_t g = 0; g < cut.P; g++)
  {
    // The verbs only read through these views; `shard<T>` carries a
    // mutable pointer because it is also the write-side descriptor.
    shards[g].data          = const_cast<T*>(data) + begin;
    shards[g].size          = sizes[g];
    shards[g].capacity      = sizes[g];
    shards[g].global_offset = begin;
    shards[g].place         = data_place::device(0);
    shards[g].exec          = group.place(g);
    shards[g].stream        = group.get_stream(g, 0);
    begin += sizes[g];
  }
  return sharded_array<T>::adopt(::std::move(shards));
}

template <class T>
sharded_array<T> adopt_cudf_column(place_group& group, const row_cut& cut, const cudf::column_view& cv)
{
  if (static_cast<::std::size_t>(cv.size()) != cut.N || cv.offset() != 0)
  {
    throw ::std::invalid_argument("adopt_cudf_column: the column must span the table's rows from offset 0");
  }
  return adopt_cudf_buffer<T>(group, cut, cv.data<T>(), cut.rows);
}

sharded_array<::std::uint32_t> adopt_cudf_null_mask(place_group& group, const row_cut& cut, const cudf::column_view& cv)
{
  if (cv.null_mask() == nullptr)
  {
    throw ::std::invalid_argument("adopt_cudf_null_mask: the column has no null mask");
  }
  return adopt_cudf_buffer<::std::uint32_t>(group, cut, cv.null_mask(), cut.words);
}

// Placed copy of an adopted column: the "born-placed" grade, filled from the
// adopted one (a shard-to-shard copy, each on its place's stream).
template <class T>
sharded_array<T>
placed_copy(place_group& group, const ::std::vector<::std::size_t>& sizes, const sharded_array<T>& adopted)
{
  auto placed = allocate_contiguous_column<T>(group, sizes);
  for (::std::size_t g = 0; g < placed.num_shards(); g++)
  {
    cuda_safe_call(cudaMemcpyAsync(
      placed.shard(g).data,
      adopted.shard(g).data,
      adopted.shard(g).size * sizeof(T),
      cudaMemcpyDeviceToDevice,
      placed.shard(g).stream));
  }
  cuda_safe_call(cudaDeviceSynchronize());
  return placed;
}

// The Q6 predicate as a BOOL8 mask column for cuDF's `apply_boolean_mask`
// (cuDF would evaluate it with `compute_column` over an AST; a tiny kernel
// keeps the example free of the AST headers and is cheaper than a chain of
// nine `binary_operation`s).
__global__ void
q6_mask_kernel(const int* shipdate, const float* discount, const int* quantity, ::std::int8_t* mask, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    mask[i] = gen::q6_selects(shipdate[i], discount[i], quantity[i]) ? 1 : 0;
  }
}

// cudaEvent timing: median of `reps` after one warm-up, the device drained
// before each start and after each stop (the verbs run on per-place lanes,
// so a single stream's events would not bracket them).
template <class Fn>
float time_median_ms(Fn&& fn, int reps = 5)
{
  cudaEvent_t start, stop;
  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));
  fn();
  ::std::vector<float> ms;
  for (int i = 0; i < reps; i++)
  {
    cuda_safe_call(cudaDeviceSynchronize());
    cuda_safe_call(cudaEventRecord(start, nullptr));
    fn();
    cuda_safe_call(cudaDeviceSynchronize());
    cuda_safe_call(cudaEventRecord(stop, nullptr));
    cuda_safe_call(cudaEventSynchronize(stop));
    float t = 0;
    cuda_safe_call(cudaEventElapsedTime(&t, start, stop));
    ms.push_back(t);
  }
  cuda_safe_call(cudaEventDestroy(start));
  cuda_safe_call(cudaEventDestroy(stop));
  ::std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}
#endif // SHARDED_RELATIONAL_WITH_CUDF
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

  const row_cut cut(N, P);

  // -------------------------------------------------------------------------
  // The table: contiguous value columns generated in place (tabulate by
  // global row id), validity words co-partitioned at rows/32.
  // -------------------------------------------------------------------------
  lineitem_table t;
  t.num_rows      = N;
  t.quantity      = allocate_contiguous_column<int>(group, cut.rows);
  t.discount      = allocate_contiguous_column<float>(group, cut.rows);
  t.extendedprice = allocate_contiguous_column<float>(group, cut.rows);
  t.shipdate      = allocate_contiguous_column<int>(group, cut.rows);
  t.group_key     = allocate_contiguous_column<int>(group, cut.rows);
  t.orderkey      = allocate_contiguous_column<int>(group, cut.rows);
  t.validity      = allocate_contiguous_column<::std::uint32_t>(group, cut.words);
  tabulate(t.quantity, envs, gen_quantity{});
  tabulate(t.discount, envs, gen_discount{});
  tabulate(t.extendedprice, envs, gen_price{});
  tabulate(t.shipdate, envs, gen_shipdate{});
  tabulate(t.group_key, envs, gen_group_key{});
  tabulate(t.orderkey, envs, gen_orderkey{});
  tabulate(t.validity, envs, gen_validity{});

  const int* orderkey_base             = t.orderkey.contiguous_data();
  const ::std::uint32_t* validity_base = t.validity.contiguous_data();

  // Row ids, materialized once: `copy_if` selects from a sharded VIEW, so
  // there is no "copy_if over a counting iterator" spelling yet (gap).
  auto row_ids = sharded_array<int>::allocate(group, cut.rows, 0);
  sequence(row_ids, envs, 0, 1);

  // A caller stream for the asynchronous terminators (`reduce_into`).
  cudaStream_t caller = nullptr;
  cuda_safe_call(cudaStreamCreateWithFlags(&caller, cudaStreamNonBlocking));
  const auto caller_prop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{caller}};
  const auto caller_env  = ::cuda::std::execution::env{caller_prop};

  bool ok = true;

  // Host references, shared by sections 1, 2 and 4.
  double ref_revenue          = 0.0;
  ::std::size_t ref_survivors = 0;
  ::std::map<int, ::std::array<double, 4>> ref_groups; // key -> {sum q, sum price, count valid, rows}
  for (::std::size_t r = 0; r < N; r++)
  {
    const float dis = gen::discount(r);
    if (gen::q6_selects(gen::shipdate(r), dis, gen::quantity(r)))
    {
      ref_revenue += static_cast<double>(gen::extendedprice(r) * dis);
      ref_survivors++;
    }
    auto& a = ref_groups[gen::group_key(r)];
    a[0] += gen::quantity(r);
    a[1] += static_cast<double>(gen::extendedprice(r));
    a[2] += gen::valid(r) ? 1 : 0;
    a[3] += 1;
  }

  // =========================================================================
  // 1. FILTER + AGGREGATE (Q6): sum(extendedprice * discount) over the rows
  //    with shipdate in a year, discount in [0.05, 0.07], quantity < 24.
  // =========================================================================
  ::std::printf("[1] filter + aggregate (Q6 shape)\n");
  auto q6_ws = make_q6_workspace(group, cut);
  {
    const q6_result r =
      run_q6(cut, envs, caller_env, caller, row_ids, t.shipdate, t.discount, t.quantity, t.extendedprice, q6_ws, true);
    const bool count_ok = (r.survivors == ref_survivors);
    const bool sum_ok =
      ::std::abs(static_cast<double>(r.revenue) - ref_revenue) <= 1e-3 * (1.0 + ::std::abs(ref_revenue));
    ok = ok && count_ok && sum_ok;
    ::std::printf(
      "  survivors %zu (ref %zu) %s, revenue %.2f (ref %.2f) %s\n",
      r.survivors,
      ref_survivors,
      count_ok ? "OK" : "MISMATCH",
      static_cast<double>(r.revenue),
      ref_revenue,
      sum_ok ? "OK" : "MISMATCH");
  }

  // =========================================================================
  // 2. GROUP-BY AGGREGATE (Q1): per group key, SUM(quantity),
  //    SUM(extendedprice), COUNT(orderkey) (non-null rows).
  // =========================================================================
  ::std::printf("[2] group-by aggregate (Q1 shape)\n");
  auto q1_ws = make_q1_workspace(group, cut);
  {
    const auto groups =
      run_q1(group, cut, envs, row_ids, t.group_key, t.quantity, t.extendedprice, t.validity, q1_ws, true);
    const bool ok2 = check_q1(groups, ref_groups, "host", 1e-3);
    ok             = ok && ok2;
    ::std::printf("  group-by vs host reference: %s\n", ok2 ? "OK" : "MISMATCH");
  }

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
  auto unmatched = sharded_array<int>::allocate(group, cut.rows, 0);
  auto nulls     = sharded_array<int>::allocate(group, cut.rows, 0);
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

#ifdef SHARDED_RELATIONAL_WITH_CUDF
  // =========================================================================
  // 4. ADOPT cuDF COLUMNS: the same two queries over real libcudf columns,
  //    adopted zero-copy, verified against libcudf's own API.
  // =========================================================================
  ::std::printf("[4] adopt cudf columns (libcudf %d.%d)\n", CUDF_VERSION_MAJOR, CUDF_VERSION_MINOR);
  {
    using cudf::data_type;
    using cudf::type_id;

    // rmm's default resource is cudaMalloc/cudaFree per allocation; route
    // cuDF's allocations (results and its own temporaries) through the
    // device's default stream-ordered pool so the timings below measure
    // cuDF's kernels rather than the allocator. The pool is still
    // whole-device memory: interleaved across the dies.
    cudaMemPool_t mempool = nullptr;
    cuda_safe_call(cudaDeviceGetDefaultMemPool(&mempool, 0));
    unsigned long long keep_all = ~0ull;
    cuda_safe_call(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &keep_all));
    rmm::mr::cuda_async_view_memory_resource pool_mr{mempool};
    rmm::mr::set_current_device_resource(cuda::mr::any_resource<cuda::mr::device_accessible>{pool_mr});
    const cudaStream_t cudf_stream = cudf::get_default_stream().value();

    // 4a. The cuDF table: libcudf's factories allocate the buffers, the
    //     generated (placed) columns are copied in once — bit-identical
    //     data, so every result of sections 1/2 is a cross-check too.
    const auto n_rows = static_cast<cudf::size_type>(N);
    auto make_col     = [&](type_id id, const void* src, ::std::size_t bytes, cudf::mask_state mask) {
      auto col = cudf::make_numeric_column(data_type{id}, n_rows, mask);
      cuda_safe_call(cudaMemcpy(col->mutable_view().head(), src, bytes, cudaMemcpyDeviceToDevice));
      return col;
    };
    ::std::vector<::std::unique_ptr<cudf::column>> cols;
    cols.push_back(
      make_col(type_id::INT32, t.quantity.contiguous_data(), N * sizeof(int), cudf::mask_state::UNALLOCATED));
    cols.push_back(
      make_col(type_id::FLOAT32, t.discount.contiguous_data(), N * sizeof(float), cudf::mask_state::UNALLOCATED));
    cols.push_back(
      make_col(type_id::FLOAT32, t.extendedprice.contiguous_data(), N * sizeof(float), cudf::mask_state::UNALLOCATED));
    cols.push_back(
      make_col(type_id::INT32, t.shipdate.contiguous_data(), N * sizeof(int), cudf::mask_state::UNALLOCATED));
    cols.push_back(
      make_col(type_id::INT32, t.group_key.contiguous_data(), N * sizeof(int), cudf::mask_state::UNALLOCATED));
    // orderkey: nullable. ALL_VALID allocates the mask; its words are then
    // overwritten with the generated validity and the null count recomputed
    // by libcudf.
    cols.push_back(
      make_col(type_id::INT32, t.orderkey.contiguous_data(), N * sizeof(int), cudf::mask_state::ALL_VALID));
    cuda_safe_call(cudaMemcpy(
      cols.back()->mutable_view().null_mask(),
      t.validity.contiguous_data(),
      (N / 32) * sizeof(::std::uint32_t),
      cudaMemcpyDeviceToDevice));
    cols.back()->set_null_count(cudf::null_count(cols.back()->view().null_mask(), 0, n_rows));
    cudf::table lineitem(::std::move(cols));
    const cudf::table_view tv           = lineitem.view();
    const cudf::column_view cv_quantity = tv.column(0), cv_discount = tv.column(1), cv_price = tv.column(2),
                            cv_shipdate = tv.column(3), cv_group_key = tv.column(4), cv_orderkey = tv.column(5);
    ::std::printf("  cudf::table: %d rows x %d columns, orderkey null_count %d (%.2f%%)\n",
                  tv.num_rows(),
                  tv.num_columns(),
                  cv_orderkey.null_count(),
                  100.0 * cv_orderkey.null_count() / N);

    // 4b. ADOPT. Nothing is copied: the shards point into cuDF's buffers.
    auto a_quantity  = adopt_cudf_column<int>(group, cut, cv_quantity);
    auto a_discount  = adopt_cudf_column<float>(group, cut, cv_discount);
    auto a_price     = adopt_cudf_column<float>(group, cut, cv_price);
    auto a_shipdate  = adopt_cudf_column<int>(group, cut, cv_shipdate);
    auto a_group_key = adopt_cudf_column<int>(group, cut, cv_group_key);
    auto a_validity  = adopt_cudf_null_mask(group, cut, cv_orderkey);
    {
      const bool alias_ok =
        a_quantity.is_view() && !a_quantity.is_owning() && a_quantity.shard(0).data == cv_quantity.data<int>()
        && a_quantity.shard(P - 1).data == cv_quantity.data<int>() + cut.row_begin[P - 1]
        && a_validity.shard(0).data == cv_orderkey.null_mask() && a_validity.size() == N / 32;
      ok = ok && alias_ok;
      ::std::printf("  adopted %zu value columns + 1 null mask as views (shards alias cudf buffers): %s\n",
                    ::std::size_t{5},
                    alias_ok ? "OK" : "MISMATCH");
      print_shard_sizes("adopted quantity (cudf data<int>())", a_quantity);
      print_shard_sizes("adopted orderkey null mask (cudf null_mask(), words)", a_validity);
    }

    // 4c. cuDF's own answers. Q6: BOOL8 mask -> apply_boolean_mask over
    //     {price, discount} -> MUL -> SUM. Q1: groupby(group_key).aggregate
    //     {SUM(quantity), SUM(price), COUNT_VALID(orderkey)}.
    auto mask_col = cudf::make_numeric_column(data_type{type_id::BOOL8}, n_rows, cudf::mask_state::UNALLOCATED);
    ::std::size_t cudf_survivors = 0;
    double cudf_revenue          = 0.0;
    auto cudf_q6                 = [&] {
      q6_mask_kernel<<<static_cast<unsigned>((N + 255) / 256), 256, 0, cudf_stream>>>(
        cv_shipdate.data<int>(),
        cv_discount.data<float>(),
        cv_quantity.data<int>(),
        mask_col->mutable_view().data<::std::int8_t>(),
        n_rows);
      cuda_safe_call(cudaGetLastError());
      auto filtered = cudf::apply_boolean_mask(cudf::table_view({cv_price, cv_discount}), mask_col->view());
      auto products = cudf::binary_operation(
        filtered->view().column(0), filtered->view().column(1), cudf::binary_operator::MUL, data_type{type_id::FLOAT32});
      auto agg       = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      auto sum       = cudf::reduce(products->view(), *agg, data_type{type_id::FLOAT64});
      cudf_survivors = static_cast<::std::size_t>(filtered->num_rows());
      cudf_revenue   = static_cast<cudf::numeric_scalar<double>*>(sum.get())->value();
    };
    ::std::map<int, ::std::array<double, 4>> cudf_groups; // key -> {sum q, sum price, count, -1 (rows unused)}
    auto cudf_q1 = [&] {
      ::std::vector<cudf::groupby::aggregation_request> requests(3);
      requests[0].values = cv_quantity;
      requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      requests[1].values = cv_price;
      requests[1].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      requests[2].values = cv_orderkey;
      requests[2].aggregations.push_back(
        cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
      cudf::groupby::groupby gb(cudf::table_view({cv_group_key}));
      auto [keys, results] = gb.aggregate(requests);
      const auto G         = static_cast<::std::size_t>(keys->num_rows());
      ::std::vector<int> h_k(G), h_c(G);
      ::std::vector<long long> h_q(G);
      ::std::vector<float> h_p(G);
      cuda_safe_call(cudaMemcpy(h_k.data(), keys->view().column(0).data<int>(), G * sizeof(int), cudaMemcpyDefault));
      cuda_safe_call(cudaMemcpy(
        h_q.data(), results[0].results[0]->view().data<long long>(), G * sizeof(long long), cudaMemcpyDefault));
      cuda_safe_call(
        cudaMemcpy(h_p.data(), results[1].results[0]->view().data<float>(), G * sizeof(float), cudaMemcpyDefault));
      cuda_safe_call(
        cudaMemcpy(h_c.data(), results[2].results[0]->view().data<int>(), G * sizeof(int), cudaMemcpyDefault));
      cudf_groups.clear();
      for (::std::size_t i = 0; i < G; i++)
      {
        cudf_groups[h_k[i]] = {
          static_cast<double>(h_q[i]), static_cast<double>(h_p[i]), static_cast<double>(h_c[i]), -1.0};
      }
    };
    cudf_q6();
    cudf_q1();
    ::std::printf(
      "  cudf Q6: survivors %zu, revenue %.2f (host ref %zu, %.2f); cudf Q1: %zu groups\n",
      cudf_survivors,
      cudf_revenue,
      ref_survivors,
      ref_revenue,
      cudf_groups.size());

    // 4d. The sharded verbs over the ADOPTED views, verified against cuDF.
    ::std::printf("  -- sharded Q6 over adopted cudf columns\n");
    const q6_result a6 =
      run_q6(cut, envs, caller_env, caller, row_ids, a_shipdate, a_discount, a_quantity, a_price, q6_ws, true);
    {
      const bool count_ok = a6.survivors == cudf_survivors;
      const bool sum_ok =
        ::std::abs(static_cast<double>(a6.revenue) - cudf_revenue) <= 1e-3 * (1.0 + ::std::abs(cudf_revenue));
      ok = ok && count_ok && sum_ok;
      ::std::printf(
        "  survivors %zu (cudf %zu) %s, revenue %.2f (cudf %.2f) %s\n",
        a6.survivors,
        cudf_survivors,
        count_ok ? "OK" : "MISMATCH",
        static_cast<double>(a6.revenue),
        cudf_revenue,
        sum_ok ? "OK" : "MISMATCH");
    }
    ::std::printf("  -- sharded Q1 over adopted cudf columns (group key, quantity, price, null mask)\n");
    {
      const auto groups = run_q1(group, cut, envs, row_ids, a_group_key, a_quantity, a_price, a_validity, q1_ws, true);
      const bool ok4    = check_q1(groups, cudf_groups, "cudf", 1e-3);
      ok                = ok && ok4;
      ::std::printf("  group-by vs cudf::groupby: %s\n", ok4 ? "OK" : "MISMATCH");
    }

    // 4e. Born-placed vs adopted: copy the adopted columns ONCE into placed
    //     (contiguous, per-place page ownership) columns and rerun. Same
    //     verbs, same cut; only where the pages live differs.
    auto p_quantity  = placed_copy(group, cut.rows, a_quantity);
    auto p_discount  = placed_copy(group, cut.rows, a_discount);
    auto p_price     = placed_copy(group, cut.rows, a_price);
    auto p_shipdate  = placed_copy(group, cut.rows, a_shipdate);
    auto p_group_key = placed_copy(group, cut.rows, a_group_key);
    auto p_validity  = placed_copy(group, cut.words, a_validity);
    {
      const auto groups = run_q1(group, cut, envs, row_ids, p_group_key, p_quantity, p_price, p_validity, q1_ws, false);
      bool same         = groups.size() == cudf_groups.size();
      for (const auto& g : groups)
      {
        const auto it = cudf_groups.find(g.key);
        same          = same && it != cudf_groups.end() && static_cast<double>(g.sum_qty) == it->second[0]
                     && static_cast<double>(g.count) == it->second[2];
      }
      ok = ok && same;
      ::std::printf("  placed copies of the adopted columns: Q1 vs cudf::groupby %s\n", same ? "OK" : "MISMATCH");
    }

    // 4f. Timing (informative only; cudaEvent, median of 5 after a warm-up).
    const float ms_cudf_q6    = time_median_ms(cudf_q6);
    const float ms_cudf_q1    = time_median_ms(cudf_q1);
    const float ms_adopted_q6 = time_median_ms([&] {
      run_q6(cut, envs, caller_env, caller, row_ids, a_shipdate, a_discount, a_quantity, a_price, q6_ws, false);
    });
    const float ms_adopted_q1 = time_median_ms([&] {
      run_q1(group, cut, envs, row_ids, a_group_key, a_quantity, a_price, a_validity, q1_ws, false);
    });
    const float ms_placed_q6  = time_median_ms([&] {
      run_q6(cut, envs, caller_env, caller, row_ids, p_shipdate, p_discount, p_quantity, p_price, q6_ws, false);
    });
    const float ms_placed_q1  = time_median_ms([&] {
      run_q1(group, cut, envs, row_ids, p_group_key, p_quantity, p_price, p_validity, q1_ws, false);
    });
    ::std::printf("  timing, N = %zu rows, ms (median of 5):\n", N);
    ::std::printf("    %-22s %10s %18s %18s\n", "query", "cudf", "sharded adopted", "sharded placed");
    ::std::printf("    %-22s %10.3f %18.3f %18.3f\n", "Q6 filter+aggregate", ms_cudf_q6, ms_adopted_q6, ms_placed_q6);
    ::std::printf("    %-22s %10.3f %18.3f %18.3f\n", "Q1 group-by", ms_cudf_q1, ms_adopted_q1, ms_placed_q1);
    ::std::printf("    (adopted = cudf's rmm memory, interleaved across the dies: confinement without placement;\n"
                  "     placed = the same bytes copied once into per-place pages; cudf = libcudf's own kernels)\n");

    // The adopted views must die before the cudf::table they alias (they do:
    // scope order), and never free anything themselves.
  }
#endif // SHARDED_RELATIONAL_WITH_CUDF

  cuda_safe_call(cudaFreeHost(q6_ws.h_revenue));
  cuda_safe_call(cudaStreamDestroy(caller));

  if (!ok)
  {
    ::std::printf("FAILED\n");
    return 1;
  }
  ::std::printf("PASSED (N=%zu, P=%zu)\n", N, P);
  return 0;
}
