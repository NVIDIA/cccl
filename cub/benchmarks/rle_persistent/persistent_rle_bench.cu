#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_run_length_encode.cuh>

#include <cuda/std/complex>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "rle_dispatch.cuh"
#include <nvbench/nvbench.cuh>

NVBENCH_DECLARE_TYPE_STRINGS(__int128_t, "I128", "__int128_t");
NVBENCH_DECLARE_TYPE_STRINGS(cuda::std::complex<float>, "C64", "cuda::std::complex<float>");

using key_types =
  nvbench::type_list<nvbench::int8_t,
                     nvbench::int16_t,
                     nvbench::int32_t,
                     nvbench::int64_t,
                     __int128_t,
                     nvbench::float32_t,
                     nvbench::float64_t,
                     cuda::std::complex<float>>;
using offset_types     = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using run_length_types = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;

namespace
{
// random keys with run lengths uniform in [1, max_seg]; consecutive runs never share a key
// (adjacency-distinctness enforced post-cast so narrow key types can't merge segments)
template <class T>
std::vector<T> gen_keys(long long n, int max_seg, unsigned seed)
{
  std::vector<T> k((size_t) n);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> seg(1, max_seg), kd(0, 1000000);
  long long i = 0;
  T prev      = T(-1);
  while (i < n)
  {
    int run = seg(rng);
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
long long cpu_run_count(const std::vector<T>& h)
{
  long long r = 0;
  for (size_t i = 0; i < h.size(); ++i)
  {
    if (i == 0 || !(h[i] == h[i - 1]))
    {
      ++r;
    }
  }
  return r;
}

template <class T, class OffsetT, class RunLengthT>
struct Bufs
{
  using NumRunsT = cub::detail::choose_signed_offset_t<OffsetT>;

  T* dk          = nullptr;
  T* dk_alloc    = nullptr;
  T* du          = nullptr;
  RunLengthT* dc = nullptr;
  NumRunsT* dn   = nullptr;
  void* dtemp    = nullptr; // persistent-RLE temp storage (tile states, cleared every launch)
  size_t tempb   = 0;
  long long n    = 0;
  long long R    = 0; // #runs
};

template <class T, class OffsetT, class RunLengthT>
Bufs<T, OffsetT, RunLengthT> setup(long long n, int max_seg, int elem_offset)
{
  using config_t = rle_impl::winner_config<T>;
  Bufs<T, OffsetT, RunLengthT> b;
  b.n                  = n;
  const size_t pad     = (size_t) ((n + config_t::kTileSize - 1) / config_t::kTileSize) * config_t::kTileSize;
  auto h               = gen_keys<T>(n, max_seg, 1u);
  const long long cpuR = cpu_run_count(h);

  cudaMalloc(&b.dk_alloc, sizeof(T) * (pad + 16));
  cudaMemset(b.dk_alloc, 0, sizeof(T) * (pad + 16));
  b.dk = b.dk_alloc + elem_offset;
  cudaMalloc(&b.du, sizeof(T) * (size_t) n);
  cudaMalloc(&b.dc, sizeof(RunLengthT) * (size_t) n);
  cudaMalloc(&b.dn, sizeof(typename Bufs<T, OffsetT, RunLengthT>::NumRunsT));
  rle_impl::persistent_rle_encode<config_t>(nullptr, b.tempb, b.dk, b.du, b.dc, b.dn, (OffsetT) n);
  cudaMalloc(&b.dtemp, b.tempb);
  cudaMemset(b.dtemp, 0xAB, b.tempb); // garbage contents; every call must clear the states itself
  cudaMemcpy(b.dk, h.data(), sizeof(T) * (size_t) n, cudaMemcpyHostToDevice);

  // Run CUB once for the authoritative run count + a correctness gate (aborts a fast-but-wrong build).
  void* tmp   = nullptr;
  size_t tbsz = 0;
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, b.dk, b.du, b.dc, b.dn, (OffsetT) n);
  cudaMalloc(&tmp, tbsz);
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, b.dk, b.du, b.dc, b.dn, (OffsetT) n);
  cudaDeviceSynchronize();
  typename Bufs<T, OffsetT, RunLengthT>::NumRunsT r_w = 0;
  cudaMemcpy(&r_w, b.dn, sizeof(r_w), cudaMemcpyDeviceToHost);
  b.R = (long long) r_w;
  cudaFree(tmp);
  if (b.R != cpuR)
  {
    std::printf("*** CUB CORRECTNESS FAIL: n=%lld max_seg=%d cub_R=%lld cpu_R=%lld ***\n", n, max_seg, b.R, cpuR);
    std::exit(3);
  }
  return b;
}

template <class T, class OffsetT, class RunLengthT>
void teardown(Bufs<T, OffsetT, RunLengthT>& b)
{
  cudaFree(b.dk_alloc);
  cudaFree(b.du);
  cudaFree(b.dc);
  cudaFree(b.dn);
  cudaFree(b.dtemp);
}

template <class T, class OffsetT, class RunLengthT>
void add_counters(nvbench::state& s, const Bufs<T, OffsetT, RunLengthT>& b)
{
  s.add_element_count(b.n);
  s.add_global_memory_reads<T>(b.n, "keys");
  s.add_global_memory_writes<T>(b.R, "unique");
  s.add_global_memory_writes<RunLengthT>(b.R, "counts");
}
} // namespace

template <class T, class OffsetT, class RunLengthT>
static void persistent_rle_bench(nvbench::state& state, nvbench::type_list<T, OffsetT, RunLengthT>)
{
  using config_t    = rle_impl::winner_config<T>;
  const long long n = state.get_int64("Elements{io}");
  const int max_seg = (int) state.get_int64("MaxSegSize");
  auto b            = setup<T, OffsetT, RunLengthT>(n, max_seg, (int) state.get_int64("Misalign"));
  add_counters(state, b);
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    rle_impl::persistent_rle_encode<config_t>(
      b.dtemp, b.tempb, b.dk, b.du, b.dc, b.dn, (OffsetT) n, launch.get_stream());
  });
  teardown(b);
}

template <class T, class OffsetT, class RunLengthT>
static void cub_rle_bench(nvbench::state& state, nvbench::type_list<T, OffsetT, RunLengthT>)
{
  const long long n = state.get_int64("Elements{io}");
  const int max_seg = (int) state.get_int64("MaxSegSize");
  auto b            = setup<T, OffsetT, RunLengthT>(n, max_seg, (int) state.get_int64("Misalign"));
  add_counters(state, b);
  void* tmp   = nullptr;
  size_t tbsz = 0;
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, b.dk, b.du, b.dc, b.dn, (OffsetT) n);
  cudaMalloc(&tmp, tbsz);
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceRunLengthEncode::Encode(tmp, tbsz, b.dk, b.du, b.dc, b.dn, (OffsetT) n, launch.get_stream());
  });
  cudaFree(tmp);
  teardown(b);
}

NVBENCH_BENCH_TYPES(persistent_rle_bench, NVBENCH_TYPE_AXES(key_types, offset_types, run_length_types))
  .set_name("persistent_rle")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "RunLengthT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {28})
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8})
  .add_int64_axis("Misalign", {0});
NVBENCH_BENCH_TYPES(cub_rle_bench, NVBENCH_TYPE_AXES(key_types, offset_types, run_length_types))
  .set_name("cub_DeviceRunLengthEncode")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "RunLengthT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {28})
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8})
  .add_int64_axis("Misalign", {0});
NVBENCH_MAIN;
