#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_run_length_encode.cuh>

#include <cuda/std/complex>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "rle_dispatch.cuh"

#ifndef K_IPT
#  define K_IPT 0
#endif

#define CHECK_CUDA(call)                                                                   \
  do                                                                                       \
  {                                                                                        \
    cudaError_t e_ = (call);                                                               \
    if (e_ != cudaSuccess)                                                                 \
    {                                                                                      \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(2);                                                                        \
    }                                                                                      \
  } while (0)

// max_seg > 0: run lengths uniform in [1, max_seg]; max_seg == 0: one constant key for the whole
// input; max_seg < 0: every run exactly -max_seg long (deterministic head positions)
template <class T>
static std::vector<T> gen_keys(long long n, int max_seg, unsigned seed)
{
  std::vector<T> k((size_t) n);
  if (max_seg == 0)
  {
    std::fill(k.begin(), k.end(), T(7));
    return k;
  }
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> seg(1, std::max(1, max_seg)), kd(0, 1000000);
  long long i = 0;
  T prev      = T(-1);
  while (i < n)
  {
    int run = (max_seg < 0) ? -max_seg : seg(rng);
    T v     = T(kd(rng));
    for (int tries = 0; v == prev && tries < 8; ++tries)
    {
      v = T(kd(rng));
    }
    prev        = v;
    long long e = std::min<long long>(i + run, n);
    for (; i < e; ++i)
    {
      k[i] = v;
    }
  }
  return k;
}

template <class T>
static double dbg(T v)
{
  if constexpr (cuda::std::is_arithmetic_v<T>)
  {
    return (double) v;
  }
  else
  {
    return 0.0; // complex/other: mismatch positions matter, values don't print
  }
}

template <class T, class OffsetT, class RunLengthT>
static bool run_case(long long n, int max_seg, unsigned seed, bool sampled = false, int elem_offset = 0)
{
  using RleConfigT = rle_impl::winner_config<T, K_IPT>;
  using NumRunsT   = cub::detail::choose_signed_offset_t<OffsetT>;

  const size_t pad = (size_t) n + elem_offset; // EXACT allocation: the bounded-TMA tail must never over-read
  auto h           = gen_keys<T>(n, max_seg, seed);

  T *dk_alloc{}, *du{};
  RunLengthT* dc{};
  NumRunsT* dn{};
  void* dtemp{};
  CHECK_CUDA(cudaMalloc(&dk_alloc, sizeof(T) * pad));
  CHECK_CUDA(cudaMemset(dk_alloc, 0, sizeof(T) * pad));
  T* const dk = dk_alloc + elem_offset;
  for (int e = 0; e < elem_offset; ++e)
  {
    CHECK_CUDA(cudaMemcpy(dk_alloc + e, h.data(), sizeof(T), cudaMemcpyHostToDevice));
  }
  CHECK_CUDA(cudaMemcpy(dk, h.data(), sizeof(T) * (size_t) n, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&du, sizeof(T) * (size_t) n));
  CHECK_CUDA(cudaMalloc(&dc, sizeof(RunLengthT) * (size_t) n));
  CHECK_CUDA(cudaMalloc(&dn, sizeof(NumRunsT)));
  size_t temp_bytes = 0;
  rle_impl::persistent_rle_encode<RleConfigT>(nullptr, temp_bytes, dk, du, dc, dn, (OffsetT) n);
  CHECK_CUDA(cudaMalloc(&dtemp, temp_bytes));
  CHECK_CUDA(cudaMemset(dtemp, 0xAB, temp_bytes)); // garbage: the per-launch clear must handle it

  // reference
  void* tmp   = nullptr;
  size_t tbsz = 0;
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, dk, du, dc, dn, (OffsetT) n);
  CHECK_CUDA(cudaMalloc(&tmp, tbsz));
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, dk, du, dc, dn, (OffsetT) n);
  CHECK_CUDA(cudaDeviceSynchronize());
  NumRunsT refR_w = -1;
  CHECK_CUDA(cudaMemcpy(&refR_w, dn, sizeof(NumRunsT), cudaMemcpyDeviceToHost));
  const long long refR = (long long) refR_w;
  // sampled mode (huge dense inputs): compare num_runs exactly + a window at each end -- the tail
  // window exercises run indices past 2^31 without 2x-full-output host mirrors
  const long long win   = sampled ? std::min<long long>(refR, 1 << 20) : refR;
  const long long tail0 = refR - win;
  std::vector<T> ref_u((size_t) win), ref_u2((size_t) (sampled ? win : 0));
  std::vector<RunLengthT> ref_c((size_t) win), ref_c2((size_t) (sampled ? win : 0));
  CHECK_CUDA(cudaMemcpy(ref_u.data(), du, sizeof(T) * win, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ref_c.data(), dc, sizeof(RunLengthT) * win, cudaMemcpyDeviceToHost));
  if (sampled)
  {
    CHECK_CUDA(cudaMemcpy(ref_u2.data(), du + tail0, sizeof(T) * win, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ref_c2.data(), dc + tail0, sizeof(RunLengthT) * win, cudaMemcpyDeviceToHost));
  }

  constexpr int kVerifyRounds = 2;
  bool ok                     = true;
  for (int round = 0; round < kVerifyRounds; ++round)
  {
    CHECK_CUDA(cudaMemset(du, 0xEE, sizeof(T) * (size_t) n));
    CHECK_CUDA(cudaMemset(dc, 0xEE, sizeof(RunLengthT) * (size_t) n));
    CHECK_CUDA(cudaMemset(dn, 0xEE, sizeof(NumRunsT)));
    CHECK_CUDA(rle_impl::persistent_rle_encode<RleConfigT>(dtemp, temp_bytes, dk, du, dc, dn, (OffsetT) n, 0));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    NumRunsT gotR_w = -1;
    CHECK_CUDA(cudaMemcpy(&gotR_w, dn, sizeof(NumRunsT), cudaMemcpyDeviceToHost));
    const long long gotR = (long long) gotR_w;
    if (gotR != refR)
    {
      std::printf("  round %d: num_runs MISMATCH: got %lld ref %lld\n", round, gotR, refR);
      ok = false;
      continue;
    }
    std::vector<T> got_u((size_t) win), got_u2((size_t) (sampled ? win : 0));
    std::vector<RunLengthT> got_c((size_t) win), got_c2((size_t) (sampled ? win : 0));
    CHECK_CUDA(cudaMemcpy(got_u.data(), du, sizeof(T) * win, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(got_c.data(), dc, sizeof(RunLengthT) * win, cudaMemcpyDeviceToHost));
    if (sampled)
    {
      CHECK_CUDA(cudaMemcpy(got_u2.data(), du + tail0, sizeof(T) * win, cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(got_c2.data(), dc + tail0, sizeof(RunLengthT) * win, cudaMemcpyDeviceToHost));
      for (long long i = 0; i < win; ++i)
      {
        if (!(got_u2[i] == ref_u2[i]) || got_c2[i] != ref_c2[i])
        {
          std::printf("  round %d TAIL run %lld mismatch\n", round, tail0 + i);
          ok = false;
          break;
        }
      }
    }
    int shown = 0;
    for (long long i = 0; i < win && shown < 5; ++i)
    {
      if (!(got_u[i] == ref_u[i]) || got_c[i] != ref_c[i])
      {
        std::printf(
          "  round %d run %lld: got (u=%g,c=%lld) ref (u=%g,c=%lld)\n",
          round,
          i,
          dbg(got_u[i]),
          (long long) got_c[i],
          dbg(ref_u[i]),
          (long long) ref_c[i]);
        ok = false;
        ++shown;
      }
    }
  }
  std::printf("%-8s n=%-12lld max_seg=%-8d off=%d runs=%-11lld\n", ok ? "PASS" : "FAIL", n, max_seg, elem_offset, refR);

  cudaFree(tmp);
  cudaFree(dk_alloc);
  cudaFree(du);
  cudaFree(dc);
  cudaFree(dn);
  cudaFree(dtemp);
  return ok;
}

template <class T, class OffsetT, class RunLengthT>
static int run_combo(const char* t_name, const char* off_name, const char* len_name, bool huge)
{
  constexpr int kTileSize = rle_impl::winner_config<T, K_IPT>::kTileSize;
  std::printf("== T=%s OffsetT=%s RunLengthT=%s (tile %d)\n", t_name, off_name, len_name, kTileSize);

  struct Case
  {
    long long n;
    int max_seg;
    int elem_offset;
  };
  const Case cases[] = {
    {200000, 2}, // mid-dispatch band at local tile geometry (caught the mid-tile-count bug)
    {150000, 3}, // mid band, different tile alignment
    {kTileSize, 1}, // single tile, dense
    {kTileSize, 1000000}, // single tile, one run
    {3 * kTileSize, 1}, // few tiles, dense
    {3 * kTileSize + 7, 1}, // partial tail tile, dense
    {3 * kTileSize + 1, 1000000}, // partial tail, open run crossing into tail
    {(1 << 20) + 12345, 2}, // partial tail, mid
    {1 << 22, 4}, // mid
    {1 << 22, 32}, // mid
    {1 << 24, 4096}, // long: head-free tiles
    {(1 << 24) + 8191, 1000000}, // very long runs + partial tail
    {1 << 28, 1}, // full bench size, dense
    {1 << 28, 1048576}, // full bench size, longest regime
    {1030 * kTileSize + 7, 1, 1}, // misaligned d_keys, dense + partial tail (above the stock-dispatch cutoff)
    {2048 * kTileSize, 32, 1}, // misaligned d_keys, mid, full last tile
    {1024ll * kTileSize, 1}, // exactly at the dispatch boundary, dense, full last tile
    {1024ll * kTileSize + 1, 1000000}, // one-element last tile, open run crossing into it
    {1024ll * kTileSize, 0}, // one constant run over the whole input: poll folds 1023 headless tiles
    {1024ll * kTileSize, -kTileSize}, // run == tile: a head at element 0 of every tile
    {1200ll * kTileSize + 3, -(kTileSize + 1)}, // run == tile+1: head position drifts through every tile offset
  };
  int fails = 0;
  for (const Case& c : cases)
  {
    for (unsigned seed : {1u, 42u})
    {
      fails += run_case<T, OffsetT, RunLengthT>(c.n, c.max_seg, seed, false, c.elem_offset) ? 0 : 1;
    }
  }
  if (huge)
  {
    if constexpr (sizeof(OffsetT) < 8)
    {
      std::printf("huge requested but OffsetT is 32-bit -- skipping\n");
    }
    else
    {
      fails += run_case<T, OffsetT, RunLengthT>((1ll << 32) + 12345, 1000000, 7u) ? 0 : 1; // element offsets past 2^31
      fails += run_case<T, OffsetT, RunLengthT>((1ll << 32), 1, 7u, /*sampled=*/true) ? 0 : 1; // 4G runs: indices past
                                                                                               // 2^31
    }
  }
  return fails;
}

template <class T>
static int run_key_type(const char* t_name, bool huge)
{
  int fails = 0;
  fails += run_combo<T, int, int>(t_name, "I32", "I32", huge);
  fails += run_combo<T, int, long long>(t_name, "I32", "I64", huge);
  fails += run_combo<T, long long, int>(t_name, "I64", "I32", huge);
  fails += run_combo<T, long long, long long>(t_name, "I64", "I64", huge);
  return fails;
}

int main(int argc, char** argv)
{
  const bool huge = (argc > 1) && argv[1][0] == 'h'; // 2^32-scale cases (wide-offset combos, big GPUs)

  int fails = 0;
  fails += run_key_type<signed char>("I8", huge);
  fails += run_key_type<short>("I16", huge);
  fails += run_key_type<int>("I32", huge);
  fails += run_key_type<long long>("I64", huge);
  fails += run_key_type<__int128_t>("I128", huge);
  fails += run_key_type<float>("F32", huge);
  fails += run_key_type<double>("F64", huge);
  fails += run_key_type<cuda::std::complex<float>>("C64", huge);

  std::printf(fails ? "*** %d FAILURES ***\n" : "ALL PASS\n", fails);
  return fails ? 1 : 0;
}
