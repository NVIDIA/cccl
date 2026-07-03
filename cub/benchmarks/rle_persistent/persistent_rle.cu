// Standalone persistent-block RLE-encode kernel (int keys) for Blackwell (B200).
// Impl only -- see persistent_rle_bench.cu for the official-nvbench comparison vs cub::DeviceRunLengthEncode.
// Types are hard-coded to int for now (keys/counts/offsets all int32).
#pragma once

#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/cstdint>

#include <algorithm> // std::min (host launcher)

#include <cuda_runtime_api.h>

namespace ptx = cuda::ptx;
using u64     = cuda::std::uint64_t;

constexpr int kIPT           = 32; // FP32s/thread; tile = kNumCompWarps*kIPT*32 = 8192
constexpr int kNumCompWarps  = 8;
constexpr int kNumStoreWarps = 16; // store warps; must divide or be a multiple of kNumCompWarps
constexpr int kStages        = 4; // pipeline depth
constexpr int kPollMlp       = 4; // how many loads each poll lane keeps in flight

// This is important for position staging on dense cases (16 way bank conflicts)
__device__ __forceinline__ int swizzle_xor_stride32(int x)
{
  return x ^ (x >> 5);
}

// CLC = 1 => use shiny new blackwell feature (UGETNEXTWORKID)
// CLC = 0 => use atomics for work stealing
// no perf difference observed on blackwell
#ifndef USE_CLC
#  define USE_CLC 1
#endif

constexpr int kWarpTileSize  = 32 * kIPT;
constexpr int kTileSize      = kNumCompWarps * kWarpTileSize;
constexpr int kNumWarps      = 1 /*load*/ + kNumCompWarps + 1 /*poll*/ + kNumStoreWarps;
constexpr int kNumThreads    = kNumWarps * 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int poll_warp_id   = 1 + kNumCompWarps;
constexpr int store_warp_id  = poll_warp_id + 1;
// for each input tile, we need to store the keys (I32) and in tile position
// for in tile position we can just do U16 since tile size is never bigger than 2^16.
constexpr size_t kDynSmem = (size_t) kStages * kTileSize * (sizeof(int) + sizeof(short));

// tile_partial_states is one 64-bit word [open_len:32][run_count:31][valid:1].
// an aligned 64-bit access is already non-tearing, but this doesn't hurt and has clear semantics
__device__ __forceinline__ void publish_state(u64* tile_state_arr, int tile_idx, int run_count, int open_len)
{
  u64 w = ((u64) (unsigned) open_len << 32) | ((u64) (unsigned) run_count << 1) | 1ull;
  cuda::atomic_ref<u64, cuda::thread_scope_device> a(tile_state_arr[tile_idx]);
  a.store(w, cuda::memory_order_relaxed);
}

// non-blocking single load of the raw packed word (valid bit may be 0)
__device__ __forceinline__ u64 load_state(u64* tile_state_arr, int tile_idx)
{
  cuda::atomic_ref<u64, cuda::thread_scope_device> a(tile_state_arr[tile_idx]);
  return a.load(cuda::memory_order_relaxed);
}

// Computes the exclusive prefix of the tile_id, i.e. the aggregate over tiles [0, tile_id)
// we do this by keeping the prefixes of the last generation and poll the partial states in
// [last_seen_tile_id, tile_id) and fold it with the aggregate we held
__device__ __forceinline__ void poll_and_fold(
  u64* tile_partial_states,
  int tile_id,
  int& last_seen_tile_id,
  int& last_seen_prefix_run_count,
  int& last_seen_prefix_open_length,
  int lane_id,
  int& curr_prefix_run_count,
  int& curr_prefix_open_length)
{
  while (last_seen_tile_id < tile_id)
  {
    const int remain = tile_id - last_seen_tile_id;
    // # of tiles to fold this iteration
    const int chunk = remain < 32 * kPollMlp ? remain : 32 * kPollMlp;
    // lane l owns the contiguous tiles
    // [last_seen_tile_id + l*kPollMlp, last_seen_tile_id + l*kPollMlp + kPollMlp)
    // clamped to `chunk`
    const int lane_base = last_seen_tile_id + lane_id * kPollMlp;
    int lane_tile_count = chunk - lane_id * kPollMlp;
    lane_tile_count     = lane_tile_count < 0 ? 0 : (lane_tile_count > kPollMlp ? kPollMlp : lane_tile_count);
    // MLP: issue all kPollMlp loads up front, then spin until this lane's owned tiles are all published.
    u64 packed_words[kPollMlp];
    bool ready;
    do
    {
      ready = true;
#pragma unroll
      for (int i = 0; i < kPollMlp; ++i)
      {
        if (i < lane_tile_count)
        {
          packed_words[i] = load_state(tile_partial_states, lane_base + i);
          if (!(packed_words[i] & 1ull))
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
        const int tile_run_count   = (int) ((packed_words[i] >> 1) & 0x7fffffffu);
        const int tile_open_length = (int) (packed_words[i] >> 32);
        lane_run_count             = lane_run_count + tile_run_count;
        lane_open_length           = (tile_run_count > 0) ? tile_open_length : (lane_open_length + tile_open_length);
      }
    }
    // cross lane fold over 32 lane aggregates
    int scan_run_count = lane_run_count, scan_open_length = lane_open_length, in_segment = (lane_run_count > 0);
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
    {
      const int pred_run_count   = __shfl_up_sync(kFullMask, scan_run_count, offset);
      const int pred_open_length = __shfl_up_sync(kFullMask, scan_open_length, offset);
      const int pred_in_segment  = __shfl_up_sync(kFullMask, in_segment, offset);
      if (lane_id >= offset)
      {
        scan_run_count += pred_run_count;
        if (!in_segment)
        {
          scan_open_length += pred_open_length;
        }
        in_segment |= pred_in_segment;
      }
    }
    // broadcast the truth from lane31
    const int chunk_run_count   = __shfl_sync(kFullMask, scan_run_count, 31);
    const int chunk_open_length = __shfl_sync(kFullMask, scan_open_length, 31);
    // combine last_seen_prefix with the chunk aggregate
    const int new_run_count = last_seen_prefix_run_count + chunk_run_count;
    const int new_open_length =
      (chunk_run_count > 0) ? chunk_open_length : (last_seen_prefix_open_length + chunk_open_length);
    last_seen_prefix_run_count   = new_run_count;
    last_seen_prefix_open_length = new_open_length;
    last_seen_tile_id += chunk;
  }
  curr_prefix_run_count   = last_seen_prefix_run_count;
  curr_prefix_open_length = last_seen_prefix_open_length;
}

// we aim for 1 block/SM since it is easier to manage resources: do not need to worry about occupancy anymore
__launch_bounds__(kNumThreads, 1) __global__ void persistent_rle(
  const int* __restrict__ d_keys,
  int* __restrict__ d_unique,
  int* __restrict__ d_counts,
  int* __restrict__ d_num_runs,
  u64* __restrict__ tile_partial_states,
#if !USE_CLC
  int* __restrict__ global_tile_counter, // global work steal counter when there is no CLC
#endif
  int num_items,
  int num_tiles)
{
  // [kStages][kTileSize] int32 ring (input keys)
  // [kStages][kTileSize] int16 staged head positions
  extern __shared__ int tile_buf[];
  short* const staged_pos = (short*) (tile_buf + (size_t) kStages * kTileSize);
  __shared__ int tile_seq[kStages]; // which global tile each ring slot holds (LOAD gets it with try_cancel)
  __shared__ int prev_tile_last_key[kStages]; // last key of the previous tile (to compute head flag for elem 0)
  __shared__ int warp_run_counts[kStages][kNumCompWarps]; // per compute warp run counts
  __shared__ int warp_first_heads[kStages][kNumCompWarps]; // per compute warp first head idx (-1 if none)
  __shared__ int warp_last_heads[kStages][kNumCompWarps]; // per compute warp last head idx (-1 if none)
  // POLL -> STORE handoff
  __shared__ int run_count_prefix[kStages]; // run count prefix
  __shared__ int open_len_prefix[kStages]; // open run len prefix

  // barriers (per ring slot):
  // LOAD --full--> COMPUTE & POLL
  // COMPUTE(all warps) --computed--> COMPUTE warp0
  // warp0 calculates & publishes this tile's aggregate to the global
  // POLL --prefixed--> STORE
  // STORE --empty--> LOAD & POLL
  __shared__ u64 full[kStages], computed[kStages], prefixed[kStages], empty[kStages];
  // COMPUTE --staged--> STORE
  __shared__ u64 staged[kStages];

#if USE_CLC
  // try_cancel writes a 16-byte response into clc_resp + completes clc_bar's tx.
  __shared__ __align__(16) uint4 clc_resp;
  __shared__ u64 clc_bar;
#endif

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
      ptx::mbarrier_init(&empty[slot_id], kNumStoreWarps);
      ptx::mbarrier_init(&staged[slot_id], kNumCompWarps); // every compute warp arrives after its scatter
    }

#if USE_CLC
    ptx::mbarrier_init(&clc_bar, 1); // 1 arrival
#endif
  }
  // normal smem writes (e.g. mbarrier_init) go through the generic proxy
  // the TMA operations access shared memory through the async proxy. these are separate visibility domains,
  // so the init writes are not automatically visible to TMA.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();

  // if you are load
  if (warp_id == 0)
  {
#if USE_CLC
    // CLC tile assignment: gen0 tile = this CTA's launch id (blockIdx.x)
    int tile_id = blk_id;
    if (lane_id == 0)
    {
      // 16 is the try_cancel byte tx
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
      ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
    }
#endif
    for (int gen = 0;; ++gen)
    {
      const int slot_id  = gen % kStages; // which slot is this?
      const int slot_gen = gen / kStages; // how many times is this slot used?
      if (gen >= kStages)
      {
        // need to wait for slot to be free
        while (!ptx::mbarrier_try_wait_parity(&empty[slot_id], (unsigned) ((slot_gen - 1) & 1)))
        {
        }
      }
#if !USE_CLC
      // work-steal: grab the next global tile via the atomic counter (one tile per atomic)
      int tile_id = 0;
      if (lane_id == 0)
      {
        tile_id = atomicAdd(global_tile_counter, 1);
      }
      tile_id = __shfl_sync(kFullMask, tile_id, 0);
#endif
      if (lane_id == 0)
      {
        tile_seq[slot_id] = tile_id;
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
      if (lane_id == 0)
      {
        // this LDG is blocking but it is ok since we just issued TMA
        prev_tile_last_key[slot_id] = (tile_id > 0) ? d_keys[(size_t) tile_id * kTileSize - 1] : 0;
        constexpr unsigned nbytes   = kTileSize * sizeof(int); // 16 KB, fits in 32 bits
        ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &full[slot_id], nbytes);
        ptx::cp_async_bulk(
          ptx::space_shared,
          ptx::space_global,
          (tile_buf + (size_t) slot_id * kTileSize),
          d_keys + (size_t) tile_id * kTileSize,
          nbytes,
          &full[slot_id]);
      }
      __syncwarp();
#if USE_CLC
      // consume the prefetched cancel
      // this is ok since it should be fast to get next cancelled id
      if (lane_id == 0)
      {
        while (!ptx::mbarrier_try_wait_parity(&clc_bar, (unsigned) (gen & 1)))
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
#endif
    }
  }
  // if you are compute
  else if (warp_id <= kNumCompWarps)
  {
    const int compute_warp_id = warp_id - 1;
    const int warp_tile_base  = compute_warp_id * kWarpTileSize;
    for (int gen = 0;; ++gen)
    {
      const int slot_id  = gen % kStages;
      const int slot_gen = gen / kStages;
      while (!ptx::mbarrier_try_wait_parity(&full[slot_id], (unsigned) (slot_gen & 1)))
      {
      }
      const int tile_id = tile_seq[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&staged[slot_id]);
        }
        break;
      }
      // slot is ready!
      const int* key_buf  = (tile_buf + (size_t) slot_id * kTileSize);
      const int tile_len  = min(kTileSize, num_items - tile_id * kTileSize);
      int local_run_count = 0, warp_first_head = -1, warp_last_head = -1;
      static_assert(kIPT <= 32, "one lane per iter requires kIPT<=32");
      // start calculating head_flags:
      // each iter is 32 consecutive elements (lane L owns loc = warp_tile_base + iter*32 + L)
      // head = (key != predecessor)
      // __ballot makes a 32-bit head mask per iter
      // the lane whose lane_id == iter stashes it, so after the loop lane L holds chunk L's mask
      short* const pos_dst = staged_pos + (size_t) slot_id * kTileSize;
      unsigned my_flags    = 0;
#pragma unroll
      for (int iter = 0; iter < kIPT; ++iter)
      {
        const int loc             = warp_tile_base + iter * 32 + lane_id;
        const int key             = (loc < tile_len) ? key_buf[loc] : 0;
        const int pred            = (loc == 0) ? prev_tile_last_key[slot_id] : key_buf[loc - 1];
        const int is_global_first = (tile_id == 0 && loc == 0);
        const int head            = (loc < tile_len) ? (is_global_first ? 1 : (key != pred)) : 0;
        const unsigned flags      = __ballot_sync(kFullMask, head);
        if (lane_id == iter)
        {
          my_flags = flags;
        }
      }
      local_run_count = __reduce_add_sync(kFullMask, __popc(my_flags));
      // each lane in a warp now has a mask that tells which chunk is non empty
      const unsigned nonempty_chunk_mask = __ballot_sync(kFullMask, my_flags != 0u);
      // if warptile is non empty (has heads), we get the location of warps first head and last head
      if (nonempty_chunk_mask)
      {
        const int first_chunk           = __ffs(nonempty_chunk_mask) - 1;
        const int last_chunk            = 31 - __clz(nonempty_chunk_mask);
        const unsigned first_chunk_mask = __shfl_sync(kFullMask, my_flags, first_chunk);
        const unsigned last_chunk_mask  = __shfl_sync(kFullMask, my_flags, last_chunk);
        warp_first_head                 = warp_tile_base + first_chunk * 32 + (__ffs(first_chunk_mask) - 1);
        warp_last_head                  = warp_tile_base + last_chunk * 32 + 31 - __clz(last_chunk_mask);
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
        {
          // kNumCompWarps<=32 so one lane/warp fits
          // (in practice we will never have anything close to 32)
          static_assert(kNumCompWarps <= 32, "insane...");
          const bool active        = lane_id < kNumCompWarps;
          const int warp_run_count = active ? warp_run_counts[slot_id][lane_id] : 0;
          const int run_count      = __reduce_add_sync(kFullMask, warp_run_count);
          // last head = the highest-index warp that has any run (its last_head is the tile's last head)
          const unsigned warps_with_runs = __ballot_sync(kFullMask, active && warp_run_count > 0);
          int last_head_idx              = -1;
          // if we have any heads, get last head index
          if (warps_with_runs)
          {
            const int last_warp_with_runs = 31 - __clz(warps_with_runs);
            last_head_idx =
              __shfl_sync(kFullMask, active ? warp_last_heads[slot_id][lane_id] : -1, last_warp_with_runs);
          }
          if (lane_id == 0)
          {
            const int open_len = (run_count > 0) ? (tile_len - last_head_idx) : tile_len;
            // CRITICAL: publish as soon as possible, this is why we calculate head_flags first
            publish_state(tile_partial_states, tile_id, run_count, open_len);
          }
        }
      }
      // now we start to calculate head positions
      {
        // we store run R at warp_tile_base + (R ^ (R>>5)) to avoid bank conflicts for dense cases
        // (CRITICAL for MaxSeg=1,2,4)
        int head_scan = __popc(my_flags); // start: this word's head count
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1)
        {
          const int pred_head_scan = __shfl_up_sync(kFullMask, head_scan, offset);
          if (lane_id >= offset)
          {
            head_scan += pred_head_scan;
          }
        }
        // head_scan is a running sum of run_count, so each lane know each chunk's base
        const int word_run_base = head_scan - __popc(my_flags);
        if (lane_id < kIPT)
        {
          unsigned pending_heads = my_flags; // this word's head mask; we need to "peel" it headbit by headbit
          const int word_pos     = warp_tile_base + lane_id * 32; // element position of bit 0 of this word
          int run_index          = word_run_base; // run-order slot for this word's next head
          while (pending_heads)
          {
            const int head_offset = __ffs(pending_heads) - 1; // offset (0..31) of the next head within the word
            pos_dst[warp_tile_base + swizzle_xor_stride32(run_index)] = (short) (word_pos + head_offset);
            ++run_index;
            pending_heads &= (pending_heads - 1); // clear the lowest set bit
          }
        }
      }
      if (lane_id == 0)
      {
        ptx::mbarrier_arrive(&staged[slot_id]); // head positions now visible to store (1/2)
      }
    }
  }
  // if you are poll
  else if (warp_id == poll_warp_id)
  {
    int last_seen_tile_id = 0, last_seen_prefix_run_count = 0, last_seen_prefix_open_length = 0;
    for (int gen = 0;; ++gen)
    {
      const int slot_id  = gen % kStages;
      const int slot_gen = gen / kStages;
      while (!ptx::mbarrier_try_wait_parity(&full[slot_id], (unsigned) (slot_gen & 1)))
      {
      }
      const int tile_id = tile_seq[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&prefixed[slot_id]); // drain
        }
        break;
      }
      int curr_prefix_run_count, curr_prefix_open_length;
      poll_and_fold(
        tile_partial_states,
        tile_id,
        last_seen_tile_id,
        last_seen_prefix_run_count,
        last_seen_prefix_open_length,
        lane_id,
        curr_prefix_run_count,
        curr_prefix_open_length);
      if (gen >= kStages) // wait for the prefix slot to drain (store done) before overwriting
      {
        while (!ptx::mbarrier_try_wait_parity(&empty[slot_id], (unsigned) ((slot_gen - 1) & 1)))
        {
        }
      }
      if (lane_id == 0)
      {
        run_count_prefix[slot_id] = curr_prefix_run_count;
        open_len_prefix[slot_id]  = curr_prefix_open_length;
        ptx::mbarrier_arrive(&prefixed[slot_id]); // prefix ready, store may proceed! (2/2)
      }
    }
  }
  // if you are store
  else
  {
    const int store_warp_idx = warp_id - store_warp_id;
    for (int gen = 0;; ++gen)
    {
      const int slot_id = gen % kStages;
      // wait for staged (1/2)
      while (!ptx::mbarrier_try_wait_parity(&staged[slot_id], (unsigned) ((gen / kStages) & 1)))
      {
      }
      const int tile_id = tile_seq[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&empty[slot_id]);
        }
        break;
      }
      const int tile_len        = min(kTileSize, num_items - tile_id * kTileSize);
      const bool is_last        = (tile_id == num_tiles - 1);
      int curr_prefix_run_count = 0, curr_prefix_open_length = 0;
      // wait for prefixed (2/2)
      while (!ptx::mbarrier_try_wait_parity(&prefixed[slot_id], (unsigned) ((gen / kStages) & 1)))
      {
      }
      curr_prefix_run_count   = run_count_prefix[slot_id];
      curr_prefix_open_length = open_len_prefix[slot_id];
      // store warps and compute warps are decoupled
      // storeW>=cw -> multiple store warps split one compute warp's runs;
      // storeW<cw -> each store warp drains cw/storeW whole regions. One must divide the other.
      static_assert(kNumStoreWarps % kNumCompWarps == 0 || kNumCompWarps % kNumStoreWarps == 0,
                    "They must divide (either direction)");
      int warp_tile_run_base[kNumCompWarps], run_base_acc = 0; // prefix of per-compute-warp run counts
#pragma unroll
      for (int warp_tile_id = 0; warp_tile_id < kNumCompWarps; ++warp_tile_id)
      {
        warp_tile_run_base[warp_tile_id] = run_base_acc;
        run_base_acc += warp_run_counts[slot_id][warp_tile_id];
      }
      const int* tile_keys = tile_buf + (size_t) slot_id * kTileSize;
      // staged positions
      const short* run_positions = staged_pos + (size_t) slot_id * kTileSize;
      // Drain runs [run_begin, run_end) of warp-tile `warp_tile_id`'s staged output into the global arrays.
      // Per run: gather its key from the run's head position -> d_unique, and write its length -> d_counts
      // (= next run's head pos - this run's head pos).
      // The warp tile's last run spans into the next warp-tile, so its length is fixed up separately.
      auto drain = [&](int warp_tile_id, int run_begin, int run_end) {
        const int warp_tile_run_count = warp_run_counts[slot_id][warp_tile_id];
        // global run index of this warp-tile's run 0 = tile's exclusive prefix + this warp-tile's base within the tile
        const int global_run_base  = curr_prefix_run_count + warp_tile_run_base[warp_tile_id];
        const int warp_tile_offset = warp_tile_id * kWarpTileSize; // this warp-tile's base in the staged arrays
        for (int run_idx = run_begin + lane_id; run_idx < run_end; run_idx += 32)
        {
          const int global_run_idx = global_run_base + run_idx;
          const int head_pos       = (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx)];
          d_unique[global_run_idx] = tile_keys[head_pos]; // gather the run's key at its head position
          if (run_idx + 1 < warp_tile_run_count)
          {
            // within-warp delta (next head - this head); the last run is fixed separately
            d_counts[global_run_idx] =
              (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)] - head_pos;
          }
        }
      };
      if constexpr (kNumStoreWarps >= kNumCompWarps)
      {
        // if we have more store warps, each warptile is split between store warps
        constexpr int kStoreWarpsPerWarpTile = kNumStoreWarps / kNumCompWarps;
        const int warp_tile_id               = store_warp_idx / kStoreWarpsPerWarpTile;
        const int sub                        = store_warp_idx % kStoreWarpsPerWarpTile;
        const int warp_tile_run_count        = warp_run_counts[slot_id][warp_tile_id];
        drain(warp_tile_id,
              (int) ((long) warp_tile_run_count * sub / kStoreWarpsPerWarpTile),
              (int) ((long) warp_tile_run_count * (sub + 1) / kStoreWarpsPerWarpTile));
      }
      else
      {
        // fewer store warps than compute regions: each store warp walks whole warptiles
        for (int warp_tile_id = store_warp_idx; warp_tile_id < kNumCompWarps; warp_tile_id += kNumStoreWarps)
        {
          drain(warp_tile_id, 0, warp_run_counts[slot_id][warp_tile_id]);
        }
      }
      // per-warp-tile boundary: a warp-tile's last run is closed by the next nonempty warp-tile's first head
      // lane L handles warp-tile L.
      if (store_warp_idx == 0 && lane_id < kNumCompWarps)
      {
        const int warp_tile_id = lane_id;
        // for warp_tile L: how many runs have you produced?
        const int warp_tile_run_count = warp_run_counts[slot_id][warp_tile_id];
        // if 0, skip: a warptile that did not start any runs has no last run to close
        if (warp_tile_run_count > 0)
        {
          // we have to close our last run. we scan L+1, L+2 for the first non-empty warptile and takes its first head
          int next_first_head = -1;
          for (int next_warp_tile = warp_tile_id + 1; next_warp_tile < kNumCompWarps; ++next_warp_tile)
          {
            if (warp_run_counts[slot_id][next_warp_tile] > 0)
            {
              next_first_head = warp_first_heads[slot_id][next_warp_tile];
              break;
            }
          }
          // now we can close it. find the corresponding global idx:
          const int last_run_global_idx =
            curr_prefix_run_count + warp_tile_run_base[warp_tile_id] + warp_tile_run_count - 1;
          if (next_first_head >= 0)
          {
            // if we found a later warptile in the tile that has runs, we only need to subtract it...
            d_counts[last_run_global_idx] = next_first_head - warp_last_heads[slot_id][warp_tile_id];
          }
          else if (is_last)
          {
            // if we are the last warptile of the whole input, we end here
            d_counts[last_run_global_idx] = tile_len - warp_last_heads[slot_id][warp_tile_id];
          }
          // else: this run is open in this tile, now this became a job for the next tile (see below)
        }
      }

      // now we need to finish last tile's open run
      if (store_warp_idx == 0 && lane_id == 0)
      {
        // this tile's first head = first head of the lowest-indexed warp with any run
        int first_head = -1;
        for (int warp_tile_id = 0; warp_tile_id < kNumCompWarps; ++warp_tile_id)
        {
          if (warp_run_counts[slot_id][warp_tile_id] > 0)
          {
            first_head = warp_first_heads[slot_id][warp_tile_id];
            break;
          }
        }
        const bool any_head = (first_head >= 0);
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

        // total run count
        if (is_last)
        {
          int total_run_count = curr_prefix_run_count;
          for (int warp_tile_id = 0; warp_tile_id < kNumCompWarps; ++warp_tile_id)
          {
            total_run_count += warp_run_counts[slot_id][warp_tile_id];
          }
          *d_num_runs = total_run_count;
        }
      }
      if (lane_id == 0)
      {
        // store done, load may proceed!
        ptx::mbarrier_arrive(&empty[slot_id]);
      }
    }
  }
}

inline void persistent_rle_launch(
  const int* d_keys,
  int* d_unique,
  int* d_counts,
  int* d_num_runs,
  u64* tile_state,
  [[maybe_unused]] int* global_tile_counter,
  int num_items,
  int num_tiles,
  cudaStream_t stream)
{
  // raise the dynamic-smem cap once (idempotent; kept off the per-launch path)
  static const bool smem_cap_set = [] {
    cudaFuncSetAttribute(persistent_rle, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kDynSmem);
    return true;
  }();
  (void) smem_cap_set;

  cudaMemsetAsync(tile_state, 0, sizeof(u64) * num_tiles, stream); // clear per-tile valid bits
#if !USE_CLC
  cudaMemsetAsync(global_tile_counter, 0, sizeof(int), stream); // reset the work-steal counter
  int numSM = 0;
  cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
  const int blocks = std::min(2 * numSM, num_tiles); // persistent work-steal grid
#else
  const int blocks = num_tiles;
#endif

  persistent_rle<<<blocks, kNumThreads, kDynSmem, stream>>>(
    d_keys,
    d_unique,
    d_counts,
    d_num_runs,
    tile_state,
#if !USE_CLC
    global_tile_counter,
#endif
    num_items,
    num_tiles);
}
