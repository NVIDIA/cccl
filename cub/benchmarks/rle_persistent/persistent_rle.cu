#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/complex>
#include <cuda/std/cstdint>

#include <cuda_runtime_api.h>

namespace rle_impl
{
namespace ptx = cuda::ptx;

template <class KeyT, int kIptOverride = 0>
struct winner_config
{
  // IPT should br 32 (32 chunks x 32 lanes)
  static constexpr int kIPT =
    (kIptOverride != 0) ? kIptOverride : ((sizeof(KeyT) >= 16) ? 8 : (sizeof(KeyT) == 8 ? 16 : 32));
  static constexpr int kNumCompWarps  = 8;
  static constexpr int kNumStoreWarps = 8; // store warps; must divide or be a multiple of kNumCompWarps
  static constexpr int kStages        = 5; // pipeline depth
  // positions ring depth: positions are written at staging and consumed by store about 2 pipeline_gens later,
  // so it can be SHALLOWER than the keys ring and this buys room for more kStages
  static constexpr int kPosBufStages = 3;
  static_assert(kPosBufStages >= 2 && kPosBufStages <= kStages, "kPosBufStages should be 2 - kStages");
  static constexpr int kPollMlp = 4; // how many loads each poll lane keeps in flight
  // when should compute warps stage?
  static constexpr int kHeadPosStagingThreshold = 64;
  // when should be pre calculate in registers?
  static constexpr int kRegBufMaxRuns      = (sizeof(KeyT) <= 4) ? 256 : (sizeof(KeyT) == 8 ? 128 : 64);
  static constexpr int kRegBufMinThreshold = 8;

  static constexpr int kWarpTileSize = 32 * kIPT;
  static constexpr int kTileSize     = kNumCompWarps * kWarpTileSize;
  static_assert(kTileSize <= 0xffff, "per-tile run_count/open_len must fit the 16-bit state-word fields");
  static constexpr int kNumWarps          = 1 /*load*/ + kNumCompWarps + 1 /*poll*/ + kNumStoreWarps + 1 /*bookkeeper*/;
  static constexpr int kNumThreads        = kNumWarps * 32;
  static constexpr int poll_warp_id       = 1 + kNumCompWarps;
  static constexpr int store_warp_id      = poll_warp_id + 1;
  static constexpr int bookkeeper_warp_id = store_warp_id + kNumStoreWarps;
  // for each input tile, we need to store the keys and in-tile positions
  // for in tile position we can just do unsigned int16 since tile size is never bigger than 2^16
  // each key slot carries kSlotPad extra leading elements
  // we overcopy one 16B chunk to the left, so that we get the last tiles boundary element
  static constexpr int kSlotPad    = 16 / sizeof(KeyT); // elements; 16 bytes = cp_async_bulk quantum
  static constexpr int kSlotStride = kTileSize + kSlotPad;
  static constexpr size_t kDynSmem =
    (size_t) kStages * kSlotStride * sizeof(KeyT) + (size_t) kPosBufStages * kTileSize * sizeof(short);
};

// This is important for position staging on dense cases (16 way bank conflicts).
__device__ __forceinline__ int swizzle_xor_stride32(int x)
{
  return x ^ (x >> 5);
}

// CLC = 1 => use shiny new blackwell feature (UGETNEXTWORKID)
// CLC = 0 => use atomics for work stealing. no perf difference observed on blackwell
// The CLC knob removed since I want to focus on blackwell perf first
// i.e. we fry one fish at a time :)

constexpr unsigned kFullMask = 0xffffffffu;

// tile_partial_states: one word per tile
// Layout: u64 [published_tag:32][open_len:16][run_count:16]
// states are cleared by rle_init_states every launch (CUB temp storage has no cross-call ownership)
// an aligned 64-bit access is already non-tearing, but atomic_ref doesn't hurt and has clear semantics
constexpr unsigned kTilePublished = 1u;

struct TilePartialStateT
{
  cuda::std::uint64_t word;

  __device__ __forceinline__ unsigned published_tag() const
  {
    return (unsigned) (word >> 32);
  }

  __device__ __forceinline__ int run_count() const
  {
    return (int) (word & 0xffffu);
  }

  __device__ __forceinline__ int open_len() const
  {
    return (int) ((word >> 16) & 0xffffu);
  }

  static __device__ __forceinline__ TilePartialStateT pack(int run_count, int open_len)
  {
    return {((cuda::std::uint64_t) kTilePublished << 32) | ((cuda::std::uint64_t) (unsigned) open_len << 16)
            | (cuda::std::uint64_t) (unsigned) run_count};
  }
};

__device__ __forceinline__ void
publish_state(TilePartialStateT* tile_state_arr, int tile_idx, int run_count, int open_len)
{
  cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device> a(tile_state_arr[tile_idx].word);
  a.store(TilePartialStateT::pack(run_count, open_len).word, cuda::memory_order_relaxed);
}

// return the state (even if not yet publish for this launch, caller checks it)
// we do not want to spin here
__device__ __forceinline__ TilePartialStateT load_state(TilePartialStateT* tile_state_arr, int tile_idx)
{
  cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device> a(tile_state_arr[tile_idx].word);
  return {a.load(cuda::memory_order_relaxed)};
}

// what is going to be the type of the prefix (run_count, open_len)?
// how do we pack them? if P is 32 bit, we compact them into 1 word. Otherwise, 2 words!
template <class OffT, bool = (sizeof(OffT) > 4)>
struct PrefixT;

template <class OffT>
struct PrefixT<OffT, false>
{
  cuda::std::uint64_t word;

  static __device__ __forceinline__ PrefixT pack(OffT run_count, OffT open_len)
  {
    return {((cuda::std::uint64_t) (unsigned) open_len << 32) | (unsigned) run_count};
  }

  __device__ __forceinline__ OffT run_count() const
  {
    return (OffT) (unsigned) (word & 0xffffffffull);
  }

  __device__ __forceinline__ OffT open_len() const
  {
    return (OffT) (unsigned) (word >> 32);
  }
};

template <class OffT>
struct alignas(16) PrefixT<OffT, true>
{
  cuda::std::uint64_t packed_run_count;
  cuda::std::uint64_t packed_open_len;

  static __device__ __forceinline__ PrefixT pack(OffT run_count, OffT open_len)
  {
    return {(cuda::std::uint64_t) run_count, (cuda::std::uint64_t) open_len};
  }

  __device__ __forceinline__ OffT run_count() const
  {
    return (OffT) packed_run_count;
  }

  __device__ __forceinline__ OffT open_len() const
  {
    return (OffT) packed_open_len;
  }
};

// position of the n-th set bit of flag_mask
// requires popc(flag_mask) > rank.
// __fns(flag_mask, 0, rank+1) computes the same thing but has NO hardware op on sm_100a and is slower
__device__ __forceinline__ int nth_set_bit(unsigned flag_mask, int rank)
{
  // each step: if the wanted bit is not among the low half's set bits, skip that half entirely
  // this is manually unrolled to reduce the count of generated SASS instructions
  int bit_position         = 0;
  int set_bits_in_low_half = __popc(flag_mask & 0xffffu);
  if (rank >= set_bits_in_low_half)
  {
    rank -= set_bits_in_low_half;
    bit_position += 16;
    flag_mask >>= 16;
  }
  set_bits_in_low_half = __popc(flag_mask & 0xffu);
  if (rank >= set_bits_in_low_half)
  {
    rank -= set_bits_in_low_half;
    bit_position += 8;
    flag_mask >>= 8;
  }
  set_bits_in_low_half = __popc(flag_mask & 0xfu);
  if (rank >= set_bits_in_low_half)
  {
    rank -= set_bits_in_low_half;
    bit_position += 4;
    flag_mask >>= 4;
  }
  set_bits_in_low_half = __popc(flag_mask & 0x3u);
  if (rank >= set_bits_in_low_half)
  {
    rank -= set_bits_in_low_half;
    bit_position += 2;
    flag_mask >>= 2;
  }
  if (rank >= (int) (flag_mask & 1u))
  {
    bit_position += 1;
  }
  return bit_position;
}

// kWidth = how many low lanes participate
template <int kWidth>
__device__ __forceinline__ int warp_inclusive_scan_add(int lane_value, int lane_id)
{
#pragma unroll
  for (int offset = 1; offset < kWidth; offset <<= 1)
  {
    const int predecessor_partial = __shfl_up_sync(kFullMask, lane_value, offset);
    if (lane_id >= offset)
    {
      lane_value += predecessor_partial;
    }
  }
  return lane_value;
}

struct WarpTileRunScanT
{
  int lane_run_count;
  int lane_runs_before;
};

template <int kNumCompWarps>
__device__ __forceinline__ WarpTileRunScanT scan_warp_tile_run_counts(const int* slot_warp_run_counts, int lane_id)
{
  const int lane_run_count = (lane_id < kNumCompWarps) ? slot_warp_run_counts[lane_id] : 0;
  const int lane_scan      = warp_inclusive_scan_add<kNumCompWarps>(lane_run_count, lane_id);
  return {lane_run_count, lane_scan - lane_run_count};
}

// start calculating head_flags:
// each iter is 32 consecutive elements (lane L owns loc = warp_tile_offset + iter*32 + L)
// head = (key != predecessor)
template <int kIPT, class KeyT>
__device__ __forceinline__ unsigned
compute_head_flags(const KeyT* key_buf, int warp_tile_offset, int tile_len, int tile_id, int lane_id)
{
  static_assert(kIPT <= 32, "one lane per iter requires kIPT<=32");
  unsigned my_flags = 0;
#pragma unroll
  for (int iter = 0; iter < kIPT; ++iter)
  {
    const int loc             = warp_tile_offset + iter * 32 + lane_id;
    const KeyT key            = (loc < tile_len) ? key_buf[loc] : KeyT{};
    const KeyT pred           = key_buf[loc - 1]; // loc==0 reads the over fetched slot[kSlotPad-1]
    const int is_global_first = (tile_id == 0 && loc == 0);
    const int head            = (loc < tile_len) ? (is_global_first ? 1 : (key != pred)) : 0;
    const unsigned flags      = __ballot_sync(kFullMask, head);
    if (lane_id == iter)
    {
      my_flags = flags;
    }
  }
  return my_flags;
}

template <int kNumCompWarps>
__device__ __forceinline__ void reduce_and_publish_tile_state(
  TilePartialStateT* tile_partial_states,
  int tile_id,
  int tile_len,
  const int* slot_warp_run_counts,
  const int* slot_warp_last_heads,
  int lane_id)
{
  // kNumCompWarps<=32 so one lane/warp fits
  // (in practice we will never have anything close to 32)
  static_assert(kNumCompWarps <= 32, "kNumCompWarps must be less than 32!");
  const bool active        = lane_id < kNumCompWarps;
  const int warp_run_count = active ? slot_warp_run_counts[lane_id] : 0;
  const int run_count      = __reduce_add_sync(kFullMask, warp_run_count);
  // last head = the highest-index warp that has any run (its last_head is the tile's last head)
  const unsigned warps_with_runs = __ballot_sync(kFullMask, active && warp_run_count > 0);
  int last_head_idx              = -1;
  // if we have any heads, get last head index
  if (warps_with_runs)
  {
    const int last_warp_with_runs = 31 - __clz(warps_with_runs);
    last_head_idx = __shfl_sync(kFullMask, active ? slot_warp_last_heads[lane_id] : -1, last_warp_with_runs);
  }
  if (lane_id == 0)
  {
    const int open_len = (run_count > 0) ? (tile_len - last_head_idx) : tile_len;
    // CRITICAL: publish as soon as possible, this is why we calculate head_flags first
    publish_state(tile_partial_states, tile_id, run_count, open_len);
  }
}

template <int kIPT>
__device__ __forceinline__ void
stage_head_positions(unsigned my_flags, short* pos_dst, int warp_tile_offset, int lane_id)
{
  // we store run R at warp_tile_offset + (R ^ (R>>5)) to avoid bank conflicts for dense cases
  // (CRITICAL for MaxSeg=1,2,4)
  int head_scan = __popc(my_flags); // start: this word's head count
  head_scan     = warp_inclusive_scan_add<32>(head_scan, lane_id);
  // head_scan is a running sum of run_count, so each lane know each chunk's base
  const int runs_before_word = head_scan - __popc(my_flags);
  if (lane_id < kIPT)
  {
    const int word_pos     = warp_tile_offset + lane_id * 32; // element position of bit 0 of this word
    unsigned pending_heads = my_flags; // this word's head mask; we need to "peel" it headbit by headbit
    int run_index          = runs_before_word; // run-order slot for this word's next head
    while (pending_heads)
    {
      const int head_offset = __ffs(pending_heads) - 1; // offset (0..31) of the next head within the word
      pos_dst[warp_tile_offset + swizzle_xor_stride32(run_index)] = (short) (word_pos + head_offset);
      ++run_index;
      pending_heads &= (pending_heads - 1); // clear the lowest set bit
    }
  }
}

struct RunSpanT
{
  int head_pos_in_warp_tile;
  int next_head_pos;
};

// the compute warp judged this warp tile too sparse to be worth the position-staging
// and it has decided to write only the 32 head-flag words
// one warp tile is 32 chunks x 32 elements so lane i owns word i
struct HeadFlagDecodeT
{
  unsigned lane_head_flag_word;
  int lane_runs_before_word;
  int lane_first_head_from_word;

  __device__ __forceinline__ HeadFlagDecodeT(const unsigned* slot_head_flags, int warp_tile_id, int lane_id)
  {
    lane_head_flag_word                = slot_head_flags[warp_tile_id * 32 + lane_id];
    const int lane_word_run_count      = __popc(lane_head_flag_word);
    const int lane_word_run_count_scan = warp_inclusive_scan_add<32>(lane_word_run_count, lane_id);
    // lane i: # of runs starting in head_flag words [0, i), i.e. in elements [0, i*32)
    lane_runs_before_word = lane_word_run_count_scan - lane_word_run_count;
    // lane i -> first head position in head flag words [i, 32)
    // if our own run_count is >0, the head is here!
    // empty should be +infinity, since we use min
    lane_first_head_from_word = lane_word_run_count ? (lane_id * 32 + __ffs(lane_head_flag_word) - 1) : 0x7fffffff;
    // if not, we loop to find the next head in flag word [i, 32). this is just a fold with min
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
    {
      const int shuffled_first_head = __shfl_down_sync(kFullMask, lane_first_head_from_word, offset);
      lane_first_head_from_word =
        min(lane_first_head_from_word, (lane_id + offset < 32) ? shuffled_first_head : 0x7fffffff);
    }
    // now, lane i holds the next head in [i, 32). we precalculate this in parallel
  }

  __device__ __forceinline__ RunSpanT decode_run(int run_idx) const
  {
    // first question: which head_flag word contains my run's (run_idx) head?
    // lane_runs_before_word's row i = number of heads in words [0, i)
    // the word containing run_dex is then the largest i with runs_before(i) that is <= j
    // we do binary search over the distributed lane_runs_before_word table held across the warp
    int flag_word_idx = 0;
#pragma unroll
    for (int step = 16; step; step >>= 1)
    {
      // propose candidate
      const int candidate_word_idx = flag_word_idx + step;
      // read the i'th row
      const int candidate_runs_before = __shfl_sync(kFullMask, lane_runs_before_word, candidate_word_idx & 31);
      if (candidate_word_idx < 32 && candidate_runs_before <= run_idx)
      {
        flag_word_idx = candidate_word_idx;
      }
    }
    // the lane knows the index of the word containing its head
    // now we need to convert it to the element position
    // where is my head in the word?
    const int run_rank_in_word = run_idx - __shfl_sync(kFullMask, lane_runs_before_word, flag_word_idx);
    // get the actual word
    const unsigned flag_word = __shfl_sync(kFullMask, lane_head_flag_word, flag_word_idx);
    // where's the first head in any word after mine?
    const int first_head_after_word = __shfl_sync(kFullMask, lane_first_head_from_word, (flag_word_idx + 1) & 31);
    // how many heads my word has?
    const int flag_word_run_count = __popc(flag_word);
    // position of my head inside the word
    const int head_bit_in_word =
      nth_set_bit(flag_word, (run_rank_in_word < flag_word_run_count) ? run_rank_in_word : 0);
    const int head_pos_in_warp_tile = flag_word_idx * 32 + head_bit_in_word;
    // where does my run end?
    // try find the position of next head in word
    const int next_head_in_word = flag_word_idx * 32 + __ffs(flag_word & (~1u << head_bit_in_word)) - 1;
    // does my word contain a head after mine? if not, next_head_in_word is garbage, and we use
    // first_head_after_word
    const int next_head_pos = (run_rank_in_word + 1 < flag_word_run_count) ? next_head_in_word : first_head_after_word;
    return {head_pos_in_warp_tile, next_head_pos};
  }
};

// drain writes [run_begin, run_end) of warp tile (warp_tile_id)'s staged output into the global arrays.
// Per run: gather its key from the run's head position -> d_unique,
// and write its length -> d_counts (= next run's head pos - this run's head pos).
// The warp tile's last run spans into the next warp-tile, so its length is fixed up separately.
template <int kHeadPosStagingThreshold, class KeyT, class LenT, class OffT>
__device__ __forceinline__ void drain_warp_tile_runs(
  KeyT* d_unique,
  LenT* d_counts,
  const KeyT* tile_keys,
  const short* run_positions,
  const unsigned* slot_head_flags,
  OffT curr_prefix_run_count,
  int warp_tile_id,
  int warp_tile_offset,
  int runs_before_warp_tile,
  int warp_tile_run_count,
  int run_begin,
  int run_end,
  int lane_id)
{
  const OffT global_runs_before_warp_tile = curr_prefix_run_count + runs_before_warp_tile;
  // this is a lot of code, but this buys us 2 - 3.5% BWUtil at MaxSegs 64 - 1M
  if (warp_tile_run_count < kHeadPosStagingThreshold)
  {
    const HeadFlagDecodeT dec(slot_head_flags, warp_tile_id, lane_id);
    // the warp process 32 runs per round ceil((run_end-run_begin)/32)
    // now each lane is assigned RUNs
    const int num_rounds = (run_end - run_begin + 31) >> 5;
    for (int it = 0; it < num_rounds; ++it)
    {
      const int run_idx  = run_begin + it * 32 + lane_id;
      const RunSpanT run = dec.decode_run(run_idx);
      if (run_idx < run_end)
      {
        const int head_pos        = warp_tile_offset + run.head_pos_in_warp_tile;
        const OffT global_run_idx = global_runs_before_warp_tile + run_idx;
        d_unique[global_run_idx]  = tile_keys[head_pos];
        if (run_idx + 1 < warp_tile_run_count)
        {
          d_counts[global_run_idx] = run.next_head_pos - run.head_pos_in_warp_tile;
        }
      }
    }
    return;
  } // if not staged
#pragma unroll 2
  // if staged
  for (int run_idx = run_begin + lane_id; run_idx < run_end; run_idx += 32)
  {
    const OffT global_run_idx = global_runs_before_warp_tile + run_idx;
    const int head_pos        = (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx)];
    d_unique[global_run_idx]  = tile_keys[head_pos]; // gather the run's key at its head position
    if (run_idx + 1 < warp_tile_run_count)
    {
      // within-warp delta (next head - this head); the last run is fixed separately
      const int run_length     = (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)] - head_pos;
      d_counts[global_run_idx] = run_length;
    }
  }
}

template <class Config, class OffT>
__device__ __forceinline__ void poll_and_fold(
  TilePartialStateT* tile_partial_states,
  int tile_id,
  int& last_seen_tile_id,
  OffT& last_seen_prefix_run_count,
  OffT& last_seen_prefix_open_length,
  int lane_id,
  OffT& curr_prefix_run_count,
  OffT& curr_prefix_open_length)
{
  constexpr int kPollMlp = Config::kPollMlp;
  while (last_seen_tile_id < tile_id)
  {
    const int remain = tile_id - last_seen_tile_id;
    // # of tiles to fold this iteration
    const int chunk = remain < 32 * kPollMlp ? remain : 32 * kPollMlp;
    // lane l owns the contiguous tiles
    // [last_seen_tile_id + l*kPollMlp, last_seen_tile_id + l*kPollMlp + kPollMlp)
    // clamped to `chunk`
    const int lane_first_tile_id = last_seen_tile_id + lane_id * kPollMlp;
    int lane_tile_count          = chunk - lane_id * kPollMlp;
    lane_tile_count              = lane_tile_count < 0 ? 0 : (lane_tile_count > kPollMlp ? kPollMlp : lane_tile_count);
    // issue all kPollMlp loads up front, then spin until this lane's owned tiles are all published (MLP)
    TilePartialStateT packed_words[kPollMlp] = {}; // must zero initialize
    bool ready;
    do
    {
      ready = true;
#pragma unroll
      for (int i = 0; i < kPollMlp; ++i)
      {
        if (i < lane_tile_count && packed_words[i].published_tag() != kTilePublished)
        {
          packed_words[i] = load_state(tile_partial_states, lane_first_tile_id + i);
          if (packed_words[i].published_tag() != kTilePublished)
          {
            ready = false;
          }
        }
      }
    } while (__ballot_sync(kFullMask, !ready) != 0u);
    // ordered reduce this lane's own tiles (increasing left -> right)
    int lane_run_count = 0, lane_open_length = 0;
#pragma unroll
    for (int i = 0; i < kPollMlp; ++i)
    {
      if (i < lane_tile_count)
      {
        const int tile_run_count   = packed_words[i].run_count();
        const int tile_open_length = packed_words[i].open_len();
        lane_run_count             = lane_run_count + tile_run_count;
        lane_open_length           = (tile_run_count > 0) ? tile_open_length : (lane_open_length + tile_open_length);
      }
    }
    // cross lane fold over 32 lane aggregates
    const int chunk_run_count      = __reduce_add_sync(kFullMask, lane_run_count);
    const unsigned lanes_with_runs = __ballot_sync(kFullMask, lane_run_count > 0);
    const int last_run_lane        = lanes_with_runs ? (31 - __clz(lanes_with_runs)) : 0;
    const int chunk_open_length    = __reduce_add_sync(kFullMask, (lane_id >= last_run_lane) ? lane_open_length : 0);
    // combine last_seen_prefix with the chunk aggregate
    const OffT new_run_count = last_seen_prefix_run_count + chunk_run_count;
    const OffT new_open_length =
      (chunk_run_count > 0) ? (OffT) chunk_open_length : (last_seen_prefix_open_length + chunk_open_length);
    last_seen_prefix_run_count   = new_run_count;
    last_seen_prefix_open_length = new_open_length;
    last_seen_tile_id += chunk;
  }
  curr_prefix_run_count   = last_seen_prefix_run_count;
  curr_prefix_open_length = last_seen_prefix_open_length;
}

// we aim for 1 block/SM since it is easier to manage resources: do not need to worry about occupancy anymore
template <class KeyT, class LenT, class NumRunsT, class OffT, class Config>
__launch_bounds__(Config::kNumThreads, 1) __global__ void persistent_rle(
  const KeyT* __restrict__ d_keys,
  KeyT* __restrict__ d_unique,
  LenT* __restrict__ d_counts,
  NumRunsT* __restrict__ d_num_runs,
  TilePartialStateT* __restrict__ tile_partial_states,
  OffT num_items,
  int num_tiles)
{
  static_assert(16 % sizeof(KeyT) == 0, "KeyT size must be a power of two <= 16");
  static_assert(alignof(KeyT) <= 16, "Alignment <= 16");
  constexpr int kIPT                     = Config::kIPT;
  constexpr int kNumCompWarps            = Config::kNumCompWarps;
  constexpr int kNumStoreWarps           = Config::kNumStoreWarps;
  constexpr int kStages                  = Config::kStages;
  constexpr int kPosBufStages            = Config::kPosBufStages;
  constexpr int kHeadPosStagingThreshold = Config::kHeadPosStagingThreshold;
  constexpr int kRegBufMinThreshold      = Config::kRegBufMinThreshold;
  constexpr int kRegBufMaxRuns           = Config::kRegBufMaxRuns;
  constexpr int kWarpTileSize            = Config::kWarpTileSize;
  constexpr int kTileSize                = Config::kTileSize;
  constexpr int kSlotPad                 = Config::kSlotPad;
  constexpr int kSlotStride              = Config::kSlotStride;
  constexpr int poll_warp_id             = Config::poll_warp_id;
  constexpr int store_warp_id            = Config::store_warp_id;
  constexpr int bookkeeper_warp_id       = Config::bookkeeper_warp_id;
  using PrefixT                          = rle_impl::PrefixT<OffT>;
  // [kStages][kTileSize] input keys
  // [kStages][kTileSize] int16 staged head positions
  extern __shared__ char smem_raw[]; // 16B-aligned; KeyT alignment <= 16 for all supported types
  KeyT* const tile_buf = (KeyT*) smem_raw;
  short* const pos_buf = (short*) (tile_buf + (size_t) kStages * kSlotStride);
  __shared__ int tile_id_buf[kStages]; // which global tile each ring slot holds (LOAD gets it with try_cancel)
  __shared__ int warp_run_counts[kStages][kNumCompWarps]; // per compute warp run counts
  __shared__ unsigned head_flag_buf[kStages][kNumCompWarps * 32]; // staged head-flag words
  __shared__ int warp_first_heads[kStages][kNumCompWarps]; // per compute warp first head idx (-1 if none)
  __shared__ int warp_last_heads[kStages][kNumCompWarps]; // per compute warp last head idx (-1 if none)

  // for POLL to pass STORE packed [open_len_prefix:32][run_count_prefix:32]
  // we need to double buffer this because STORE will arrive empty immediately after reading the data
  // i.e. when POLL start to write g+kStage's prefix data, STORE and BOOKKEEPER might still have not read
  // the prefix data. therefore, we MUST to double buffer this to ensure it is safe.
  // the proof is as follows: with double buffering, the hazard becomes when POLL for g+2*kStage start to write,
  // could STORE and BOOKKEEPER still wait on the prefix data of g+kStage? This CANNOT happen, because for us to
  // start loading g+2*kStage, STORE and BOOKKEEPER must have all arrived empty for g+kStage, which means they have
  // fully processed the data for g.
  __shared__ PrefixT prefix_packed[kStages][2];

  // STORE --pos_buf_free--> COMPUTE staging (only when kPosBufStages < kStages; if = then it is mapped 1:1)
  // all store warps arrive after they no longer need the data in the pos slot; compute waits before re-staging into it
  __shared__ cuda::std::uint64_t pos_buf_free[kPosBufStages];
  // LOAD --full--> COMPUTE & POLL
  // COMPUTE(all warps) --computed--> COMPUTE warp0
  // warp0 calculates & publishes this tile's aggregate to the global
  // POLL --prefixed--> STORE
  // STORE --empty--> LOAD & POLL
  __shared__ cuda::std::uint64_t full[kStages];
  __shared__ cuda::std::uint64_t computed[kStages], prefixed[kStages], empty[kStages];
  // COMPUTE warp w --staged_warp_tile[w]--> STORE: we arrive per warp tile handoff
  // i.e. store warps start working to drain a warp-tile as soon as ITS positions are staged
  // instead of waiting for all 8 compute warps (warp 0 is always slower!!)
  // The shared metadata store also needs (run counts, first/last heads, tile_id_buf) is covered by `computed`.
  __shared__ cuda::std::uint64_t staged_warp_tile[kStages][kNumCompWarps];

  // try_cancel writes a 16-byte response into clc_resp + completes clc_bar's tx.
  __shared__ __align__(16) uint4 clc_resp;
  __shared__ cuda::std::uint64_t clc_bar;

  const int thr_id  = threadIdx.x;
  const int warp_id = thr_id >> 5;
  const int lane_id = thr_id & 31;
  const int blk_id  = blockIdx.x;
  if (thr_id == 0)
  {
    for (int slot_id = 0; slot_id < kStages; ++slot_id)
    {
      ptx::mbarrier_init(&full[slot_id], 1);
      ptx::mbarrier_init(&computed[slot_id], kNumCompWarps); // every compute warp arrives
      ptx::mbarrier_init(&prefixed[slot_id], 1);
      ptx::mbarrier_init(&empty[slot_id], kNumStoreWarps + 1); // store warps + the bookkeeper
      for (int cw = 0; cw < kNumCompWarps; ++cw)
      {
        ptx::mbarrier_init(&staged_warp_tile[slot_id][cw], 1); // that compute warp's lane0, after its scatter
      }
    }
    for (int p = 0; p < kPosBufStages; ++p)
    {
      ptx::mbarrier_init(&pos_buf_free[p], kNumStoreWarps);
    }

    ptx::mbarrier_init(&clc_bar, 1); // 1 arrival
  }
  // normal smem writes (e.g. mbarrier_init) go through the generic proxy
  // the TMA operations access shared memory through the async proxy. these are separate visibility domains,
  // so the init writes are not automatically visible to TMA.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();

  // if you are load
  if (warp_id == 0)
  {
    // CLC tile assignment: gen0 tile = this CTA's launch id (blockIdx.x)
    int tile_id = blk_id;
    if (lane_id == 0)
    {
      // 16 is the try_cancel byte tx
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
      ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
    }
    for (int pipeline_gen = 0;; ++pipeline_gen)
    {
      const int slot_id  = pipeline_gen % kStages; // which slot is this?
      const int slot_gen = pipeline_gen / kStages; // how many times is this slot used?
      if (pipeline_gen >= kStages)
      {
        // need to wait for slot to be free
        while (!ptx::mbarrier_try_wait_parity(&empty[slot_id], (unsigned) ((slot_gen - 1) & 1)))
        {
        }
      }
      if (lane_id == 0)
      {
        tile_id_buf[slot_id] = tile_id;
      }
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&full[slot_id]);
        }
        __syncwarp();
        break;
      }
      // over-fetch one 16B chunk to the left, so that we get last tiles last key
      // tile 0 has no predecessor and skips the over-fetch
      const bool first_tile = (tile_id == 0);
      const int tile_len    = (int) min((OffT) kTileSize, num_items - (OffT) tile_id * kTileSize);
      if (lane_id == 0)
      {
        const unsigned nbytes = (unsigned) (((size_t) tile_len + (first_tile ? 0 : kSlotPad)) * sizeof(KeyT));
        if (tile_len == kTileSize)
        {
          ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &full[slot_id], nbytes);
          ptx::cp_async_bulk(
            ptx::space_shared,
            ptx::space_global,
            tile_buf + (size_t) slot_id * kSlotStride + (first_tile ? kSlotPad : 0),
            d_keys + (size_t) tile_id * kTileSize - (first_tile ? 0 : kSlotPad),
            nbytes,
            &full[slot_id]);
        }
        else
        {
          const unsigned span_bytes = (nbytes + 15u) & ~15u;
          ptx::mbarrier_arrive_expect_tx(
            ptx::sem_release, ptx::scope_cta, ptx::space_shared, &full[slot_id], span_bytes);
          ptx::cp_async_bulk_ignore_oob(
            ptx::space_shared,
            ptx::space_global,
            tile_buf + (size_t) slot_id * kSlotStride + (first_tile ? kSlotPad : 0),
            d_keys + (size_t) tile_id * kTileSize - (first_tile ? 0 : kSlotPad),
            span_bytes,
            0u,
            span_bytes - nbytes,
            &full[slot_id]);
        }
      }
      __syncwarp();
      // consume the prefetched cancel
      // this is ok since it should be fast to get next cancelled id
      if (lane_id == 0)
      {
        while (!ptx::mbarrier_try_wait_parity(&clc_bar, (unsigned) (pipeline_gen & 1)))
        {
        }
        // try_cancel wrote clc_resp via the async proxy
        ptx::fence_proxy_async(ptx::space_shared);
        const bool canceled = ptx::clusterlaunchcontrol_query_cancel_is_canceled(clc_resp);
        int nxt             = num_tiles; // if no more work was cancellable
        if (canceled)
        {
          nxt = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(clc_resp);
          ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
          ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
        }
        tile_id = nxt;
      }
      tile_id = __shfl_sync(kFullMask, tile_id, 0);
    }
  }
  // if you are compute
  else if (warp_id <= kNumCompWarps)
  {
    const int compute_warp_id  = warp_id - 1;
    const int warp_tile_offset = compute_warp_id * kWarpTileSize;
    for (int pipeline_gen = 0;; ++pipeline_gen)
    {
      const int slot_id  = pipeline_gen % kStages;
      const int slot_gen = pipeline_gen / kStages;
      while (!ptx::mbarrier_try_wait_parity(&full[slot_id], (unsigned) (slot_gen & 1)))
      {
      }
      const int tile_id = tile_id_buf[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          // STORE waits computed + its warp-tile's staged_warp_tile, so arrive both
          ptx::mbarrier_arrive(&computed[slot_id]);
          ptx::mbarrier_arrive(&staged_warp_tile[slot_id][compute_warp_id]);
        }
        break;
      }
      // slot is ready!
      const KeyT* key_buf = tile_buf + (size_t) slot_id * kSlotStride + kSlotPad;
      const int tile_len  = (int) min((OffT) kTileSize, num_items - (OffT) tile_id * kTileSize);
      int local_run_count = 0, warp_first_head = -1, warp_last_head = -1;
      short* const pos_dst    = pos_buf + (size_t) (pipeline_gen % kPosBufStages) * kTileSize;
      const unsigned my_flags = compute_head_flags<kIPT>(key_buf, warp_tile_offset, tile_len, tile_id, lane_id);
      local_run_count         = __reduce_add_sync(kFullMask, __popc(my_flags));
      // each lane in a warp now has a mask that tells which chunk is non empty
      const unsigned nonempty_chunk_mask = __ballot_sync(kFullMask, my_flags != 0u);
      // if warptile is non empty (has heads), we get the location of warps first head and last head
      if (nonempty_chunk_mask)
      {
        const int first_chunk           = __ffs(nonempty_chunk_mask) - 1;
        const int last_chunk            = 31 - __clz(nonempty_chunk_mask);
        const unsigned first_chunk_mask = __shfl_sync(kFullMask, my_flags, first_chunk);
        const unsigned last_chunk_mask  = __shfl_sync(kFullMask, my_flags, last_chunk);
        warp_first_head                 = warp_tile_offset + first_chunk * 32 + (__ffs(first_chunk_mask) - 1);
        warp_last_head                  = warp_tile_offset + last_chunk * 32 + 31 - __clz(last_chunk_mask);
      }
      // now, we calculate warptile aggregates
      if (lane_id == 0)
      {
        warp_run_counts[slot_id][compute_warp_id]  = local_run_count;
        warp_first_heads[slot_id][compute_warp_id] = warp_first_head;
        warp_last_heads[slot_id][compute_warp_id]  = warp_last_head;
        ptx::mbarrier_arrive(&computed[slot_id]); // each compute warp arrives
      }
      // warp 0 waits all compute warp arrivals so that every warp's results are visible
      // then collect results from all warptiles and publish the tile run count and tile open len
      if (compute_warp_id == 0)
      {
        while (!ptx::mbarrier_try_wait_parity(&computed[slot_id], (unsigned) (slot_gen & 1)))
        {
        }
        reduce_and_publish_tile_state<kNumCompWarps>(
          tile_partial_states, tile_id, tile_len, warp_run_counts[slot_id], warp_last_heads[slot_id], lane_id);
      }
      // now we start to stage head positions per warp tile, if a warptile has enough runs
      // (it is only worth it when we have more runs by a certain threshold per warp tile)
      // (otherwise, it is cheaper to recalculate positions from head_flags directly)
      const bool stage_flags = (local_run_count < kHeadPosStagingThreshold);
      if (stage_flags)
      {
        head_flag_buf[slot_id][compute_warp_id * 32 + lane_id] = my_flags;
      }
      else
      {
        if constexpr (kPosBufStages < kStages)
        {
          // the pos slot is shared by pipeline_gens g, g+kPosBufStages, ...
          // need to wait for it to be cleared by STORE
          if (pipeline_gen >= kPosBufStages)
          {
            while (!ptx::mbarrier_try_wait_parity(
              &pos_buf_free[pipeline_gen % kPosBufStages], (unsigned) ((pipeline_gen / kPosBufStages - 1) & 1)))
            {
            }
          }
        }
        stage_head_positions<kIPT>(my_flags, pos_dst, warp_tile_offset, lane_id);
      } // stage flags
      if (lane_id == 0)
      {
        ptx::mbarrier_arrive(&staged_warp_tile[slot_id][compute_warp_id]); // this warp-tile's positions ready
      }
    }
  }
  // if you are poll
  else if (warp_id == poll_warp_id)
  {
    int last_seen_tile_id             = 0;
    OffT last_seen_prefix_run_count   = 0;
    OffT last_seen_prefix_open_length = 0;
    for (int pipeline_gen = 0;; ++pipeline_gen)
    {
      const int slot_id  = pipeline_gen % kStages;
      const int slot_gen = pipeline_gen / kStages;
      while (!ptx::mbarrier_try_wait_parity(&full[slot_id], (unsigned) (slot_gen & 1)))
      {
      }
      const int tile_id = tile_id_buf[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&prefixed[slot_id]);
        }
        break;
      }
      OffT curr_prefix_run_count, curr_prefix_open_length;
      poll_and_fold<Config>(
        tile_partial_states,
        tile_id,
        last_seen_tile_id,
        last_seen_prefix_run_count,
        last_seen_prefix_open_length,
        lane_id,
        curr_prefix_run_count,
        curr_prefix_open_length);
      // no wait needed before overwriting the prefix slot since we can prove this is safe with double buffering
      // (proof see above at barrier initiation)
      if (lane_id == 0)
      {
        prefix_packed[slot_id][slot_gen & 1] = PrefixT::pack(curr_prefix_run_count, curr_prefix_open_length);
        ptx::mbarrier_arrive(&prefixed[slot_id]); // prefix ready, store may proceed
      }
    }
  }
  // if you are store
  else if (warp_id < bookkeeper_warp_id)
  {
    const int store_warp_idx = warp_id - store_warp_id;
    for (int pipeline_gen = 0;; ++pipeline_gen)
    {
      const int slot_id = pipeline_gen % kStages;
      // wait for computed (1/3): all per-warp-tile metadata (run counts, first/last heads)
      while (!ptx::mbarrier_try_wait_parity(&computed[slot_id], (unsigned) ((pipeline_gen / kStages) & 1)))
      {
      }
      const int tile_id = tile_id_buf[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&empty[slot_id]);
        }
        break;
      }
      // store warps and compute warps are decoupled
      // fewer store warps than compute warps has no winning regime since warp slots are not scarce at 1 block/SM
      static_assert(kNumStoreWarps >= kNumCompWarps && kNumStoreWarps % kNumCompWarps == 0,
                    "store warps: a whole multiple of compute warps");
      // per-warp-tile run bases (lane i owns warp-tile i's count/base) and done BEFORE the wait on prefixed so they
      // overlap
      // lane i: run-count sum over warp-tiles [0, i) = where warp-tile i's runs begin within the tile
      const auto [lane_warp_tile_run_count, lane_runs_before_warp_tile] =
        scan_warp_tile_run_counts<kNumCompWarps>(warp_run_counts[slot_id], lane_id);
      const KeyT* tile_keys = tile_buf + (size_t) slot_id * kSlotStride + kSlotPad;
      // staged positions
      const short* run_positions = pos_buf + (size_t) (pipeline_gen % kPosBufStages) * kTileSize;
      // wait for prefixed (2/3)
      auto wait_prefixed_and_read = [&]() {
        while (!ptx::mbarrier_try_wait_parity(&prefixed[slot_id], (unsigned) ((pipeline_gen / kStages) & 1)))
        {
        }
        return prefix_packed[slot_id][(pipeline_gen / kStages) & 1].run_count();
      };
      // since we have more store warps, each warptile is split between store warps
      constexpr int kStoreWarpsPerWarpTile = kNumStoreWarps / kNumCompWarps;
      const int warp_tile_id               = store_warp_idx / kStoreWarpsPerWarpTile;
      const int sub                        = store_warp_idx % kStoreWarpsPerWarpTile;
      const int warp_tile_run_count        = __shfl_sync(kFullMask, lane_warp_tile_run_count, warp_tile_id);
      const int runs_before_warp_tile      = __shfl_sync(kFullMask, lane_runs_before_warp_tile, warp_tile_id);
      // if our register budget allows it and it is worth it, we can buffer intermediate results in register
      // and arrive empty early. this buys 2.5% BWUtil at the worst segments
      if (warp_tile_run_count >= kRegBufMinThreshold && warp_tile_run_count <= kRegBufMaxRuns)
      {
        const int run_begin = (int) ((long) warp_tile_run_count * sub / kStoreWarpsPerWarpTile);
        const int run_end   = (int) ((long) warp_tile_run_count * (sub + 1) / kStoreWarpsPerWarpTile);
        // wait for staged_warp_tile (3/3)
        while (!ptx::mbarrier_try_wait_parity(
          &staged_warp_tile[slot_id][warp_tile_id], (unsigned) ((pipeline_gen / kStages) & 1)))
        {
        }
        constexpr int kBufPerLane = ((kRegBufMaxRuns + 31) / 32 > 0) ? (kRegBufMaxRuns + 31) / 32 : 1;
        KeyT buf_key[kBufPerLane];
        int buf_run_length[kBufPerLane];
        const int warp_tile_offset = warp_tile_id * kWarpTileSize;
        const int num_rounds       = (run_end - run_begin + 31) >> 5;
        if (warp_tile_run_count < kHeadPosStagingThreshold)
        {
          const HeadFlagDecodeT dec(head_flag_buf[slot_id], warp_tile_id, lane_id);
#pragma unroll
          for (int it = 0; it < kBufPerLane; ++it)
          {
            if (it >= num_rounds)
            {
              break;
            }
            const int run_idx  = run_begin + it * 32 + lane_id;
            const RunSpanT run = dec.decode_run(run_idx);
            buf_key[it]        = (run_idx < run_end) ? tile_keys[warp_tile_offset + run.head_pos_in_warp_tile] : KeyT{};
            buf_run_length[it] = run.next_head_pos - run.head_pos_in_warp_tile;
          }
        } // if not staged
        else
        {
#pragma unroll
          for (int it = 0; it < kBufPerLane; ++it)
          {
            if (it >= num_rounds)
            {
              break;
            }
            const int run_idx  = run_begin + it * 32 + lane_id;
            const bool act     = run_idx < run_end;
            const int head_pos = act ? (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx)] : 0;
            buf_key[it]        = tile_keys[head_pos];
            buf_run_length[it] = (act && run_idx + 1 < warp_tile_run_count)
                                 ? (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)] - head_pos
                                 : 0;
          }
        }
        if (lane_id == 0)
        {
          if constexpr (kPosBufStages < kStages)
          {
            ptx::mbarrier_arrive(&pos_buf_free[pipeline_gen % kPosBufStages]);
          }
          ptx::mbarrier_arrive(&empty[slot_id]); // with register buffers we can arrive early
        }
        const OffT global_runs_before_warp_tile = wait_prefixed_and_read() + runs_before_warp_tile;
#pragma unroll
        for (int it = 0; it < kBufPerLane; ++it)
        {
          if (it >= num_rounds)
          {
            break;
          }
          const int run_idx = run_begin + it * 32 + lane_id;
          if (run_idx < run_end)
          {
            const OffT global_run_idx = global_runs_before_warp_tile + run_idx;
            d_unique[global_run_idx]  = buf_key[it];
            if (run_idx + 1 < warp_tile_run_count)
            {
              d_counts[global_run_idx] = buf_run_length[it];
            }
          }
        }
        continue;
      } // reg buf
      // if not reg buffed, we do the normal things, i.e. prefixed wait, then staged_warp_tile, then drain
      const OffT curr_prefix_run_count = wait_prefixed_and_read();
      // wait for staged_warp_tile (3/3)
      while (!ptx::mbarrier_try_wait_parity(
        &staged_warp_tile[slot_id][warp_tile_id], (unsigned) ((pipeline_gen / kStages) & 1)))
      {
      }
      drain_warp_tile_runs<kHeadPosStagingThreshold>(
        d_unique,
        d_counts,
        tile_keys,
        run_positions,
        head_flag_buf[slot_id],
        curr_prefix_run_count,
        warp_tile_id,
        warp_tile_id * kWarpTileSize,
        runs_before_warp_tile,
        warp_tile_run_count,
        (int) ((long) warp_tile_run_count * sub / kStoreWarpsPerWarpTile),
        (int) ((long) warp_tile_run_count * (sub + 1) / kStoreWarpsPerWarpTile),
        lane_id);
      if (lane_id == 0)
      {
        if constexpr (kPosBufStages < kStages)
        {
          ptx::mbarrier_arrive(&pos_buf_free[pipeline_gen % kPosBufStages]);
        }
        // store done, load may proceed!
        ptx::mbarrier_arrive(&empty[slot_id]);
      }
    }
  }
  // if you are the bookkeeper (i should rename this to boundarycloser...)
  else
  {
    for (int pipeline_gen = 0;; ++pipeline_gen)
    {
      const int slot_id = pipeline_gen % kStages;
      while (!ptx::mbarrier_try_wait_parity(&computed[slot_id], (unsigned) ((pipeline_gen / kStages) & 1)))
      {
      }
      const int tile_id = tile_id_buf[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&empty[slot_id]);
        }
        break;
      }
      const int tile_len = (int) min((OffT) kTileSize, num_items - (OffT) tile_id * kTileSize);
      const bool is_last = (tile_id == num_tiles - 1);
      // same scan as the store warps (lane i = warp-tile i)
      const auto [lane_warp_tile_run_count, lane_runs_before_warp_tile] =
        scan_warp_tile_run_counts<kNumCompWarps>(warp_run_counts[slot_id], lane_id);
      const int tile_total_runs =
        __shfl_sync(kFullMask, lane_runs_before_warp_tile + lane_warp_tile_run_count, kNumCompWarps - 1);
      const unsigned nonempty_warp_tiles_mask = __ballot_sync(kFullMask, lane_warp_tile_run_count > 0);
      while (!ptx::mbarrier_try_wait_parity(&prefixed[slot_id], (unsigned) ((pipeline_gen / kStages) & 1)))
      {
      }
      const PrefixT packed_prefix        = prefix_packed[slot_id][(pipeline_gen / kStages) & 1];
      const OffT curr_prefix_run_count   = packed_prefix.run_count();
      const OffT curr_prefix_open_length = packed_prefix.open_len();
      // per-warp-tile boundary: a warp-tile's last run is closed by the next nonempty warp-tile's
      // first head. lane L handles warp-tile L.
      if (lane_id < kNumCompWarps && lane_warp_tile_run_count > 0)
      {
        const unsigned later_wts = nonempty_warp_tiles_mask >> (lane_id + 1); // nonempty warp-tiles after L
        const OffT last_run_global_idx =
          curr_prefix_run_count + lane_runs_before_warp_tile + lane_warp_tile_run_count - 1;
        if (later_wts)
        {
          const int next_wt             = lane_id + 1 + __ffs(later_wts) - 1;
          d_counts[last_run_global_idx] = warp_first_heads[slot_id][next_wt] - warp_last_heads[slot_id][lane_id];
        }
        else if (is_last)
        {
          // if we are the last warptile of the whole input, we end here
          d_counts[last_run_global_idx] = tile_len - warp_last_heads[slot_id][lane_id];
        }
        // else: this run is open in this tile, now this became a job for the next tile (see below)
      }
      // now we need to finish last tile's open run
      if (lane_id == 0)
      {
        const bool any_head  = (nonempty_warp_tiles_mask != 0);
        const int first_head = any_head ? warp_first_heads[slot_id][__ffs(nonempty_warp_tiles_mask) - 1] : -1;
        // if our tile has a head, i.e. it stops here
        if (any_head && curr_prefix_run_count > 0)
        {
          d_counts[curr_prefix_run_count - 1] = curr_prefix_open_length + first_head;
        }
        // if we are last tile with no head: we have to close it here
        if (is_last && !any_head && curr_prefix_run_count > 0)
        {
          d_counts[curr_prefix_run_count - 1] = curr_prefix_open_length + tile_len;
        }
        // otherwise, next tile's problem
        if (is_last)
        {
          *d_num_runs = (NumRunsT) (curr_prefix_run_count + tile_total_runs);
        }
        ptx::mbarrier_arrive(&empty[slot_id]); // bookkeeping done, slot may recycle
      }
    }
  }
}

template <class KeyT, class LenT, class NumRunsT, class OffT, class Config = winner_config<KeyT>>
inline cudaError_t persistent_rle_launch(
  const KeyT* d_keys,
  KeyT* d_unique,
  LenT* d_counts,
  NumRunsT* d_num_runs,
  TilePartialStateT* tile_state,
  OffT num_items,
  int num_tiles,
  cudaStream_t stream)
{
  constexpr size_t kDynSmem = Config::kDynSmem;
  constexpr int kNumThreads = Config::kNumThreads;

  const cudaError_t error = cudaFuncSetAttribute(
    persistent_rle<KeyT, LenT, NumRunsT, OffT, Config>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kDynSmem);
  if (error != cudaSuccess)
  {
    return error;
  }

  const int blocks = num_tiles;

  persistent_rle<KeyT, LenT, NumRunsT, OffT, Config><<<blocks, kNumThreads, kDynSmem, stream>>>(
    d_keys, d_unique, d_counts, d_num_runs, tile_state, num_items, num_tiles);
  return cudaPeekAtLastError();
}
} // namespace rle_impl
