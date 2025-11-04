// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_transform.cuh>

#include <look_back_helper.cuh>
#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5
// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1

#if !TUNE_BASE
#  if TUNE_TRANSPOSE == 0
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_DIRECT
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_WARP_TRANSPOSE
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using ScanByKeyPolicyT = cub::AgentScanByKeyPolicy<
      TUNE_THREADS,
      TUNE_ITEMS,
      // TODO Tune
      TUNE_LOAD_ALGORITHM,
      TUNE_LOAD_MODIFIER,
      cub::BLOCK_SCAN_WARP_SCANS,
      TUNE_STORE_ALGORITHM,
      delay_constructor_t>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

namespace impl
{
/* Consider free monoid with two generators, ``q`` and ``p``, modulo defining relationship (``p * q == 1``).
 * Elements of this algebra are ``q^m * p^n``, identified by a pair of integral exponents. The identity
 * element is ``1 == q^0 * p^0``, which maps to pair of zeros ``e = (0, 0)``.
 *
 * The product is defined by concatenation:
 *     q^m * p^n * q^r * p^s == q^m * p^{n-1} * p * q * q^{r-1} * p^s
 *                           == q^m * p^{n-1} * q^{r-1} * p^s
 *
 * This reduction can be performed ``min(n, r)`` times resulting in
 *
 *     q^m * p^n * q^r * p^s == q^{m + r - min(n, r)} * p^{s + n - min(n, r)}
 *
 * Hence this is a monoid, known as bicyclic monoid.
 * This operation of pairs of integers is associative (since concatenation is), but non-commutative.
 *
 *    Ref: https://en.wikipedia.org/wiki/Bicyclic_semigroup
 *    Ref: https://en.wikipedia.org/wiki/Monoid
 */

template <typename UnsignedIntegralT>
struct bicyclic_monoid_op
{
  static_assert(cuda::std::is_integral_v<UnsignedIntegralT>);
  static_assert(cuda::std::is_unsigned_v<UnsignedIntegralT>);

  using pair_t = cuda::std::pair<UnsignedIntegralT, UnsignedIntegralT>;
  using min_t  = cuda::minimum<>;

  // Operator is associative but non-commutative
  pair_t __host__ __device__ operator()(pair_t v1, pair_t v2) const
  {
    auto [m, n] = v1;
    auto [r, s] = v2;
    auto min_nr = min_t{}(n, r);
    return {m + r - min_nr, s + n - min_nr};
  }
};

template <typename T>
struct repack_pair
{
  cuda::std::pair<T, T> __host__ __device__ operator()(const T& v1, const T& v2) const
  {
    return {v1, v2};
  };
};
}; // namespace impl

template <typename T, typename OffsetT>
static void inclusive_scan(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  static_assert(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>, "Unsigned integral type should be used");
  using wrapped_init_t = cub::NullType;
  using pair_t         = cuda::std::pair<T, T>;
  using op_t           = impl::bicyclic_monoid_op<T>;
  using accum_t        = pair_t;
  using input_it_t     = const pair_t*;
  using output_it_t    = pair_t*;
  using offset_t       = cub::detail::choose_offset_t<OffsetT>;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<accum_t>;
  using dispatch_t = cub::
    DispatchScan<input_it_t, output_it_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No, policy_t>;
#else
  using dispatch_t =
    cub::DispatchScan<input_it_t, output_it_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No>;
#endif

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<pair_t> output(elements);

  thrust::device_vector<pair_t> input(elements);
  {
    thrust::device_vector<T> q_exponents = generate(elements);
    thrust::device_vector<T> p_exponents = generate(elements);

    impl::repack_pair<T> repack_op{};

    cub::DeviceTransform::Transform(
      cuda::std::tuple{q_exponents.begin(), p_exponents.begin()}, input.begin(), elements, repack_op);

    // deallocate temporary arrays at the scope boundary
  }

  pair_t* d_input  = thrust::raw_pointer_cast(input.data());
  pair_t* d_output = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<pair_t>(elements, "Size");
  state.add_global_memory_writes<pair_t>(elements);

  cudaStream_t bench_stream = state.get_cuda_stream();

  size_t tmp_size;
  dispatch_t::Dispatch(nullptr, tmp_size, d_input, d_output, op_t{}, wrapped_init_t{}, input.size(), bench_stream);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);
  nvbench::uint8_t* d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      d_tmp, tmp_size, d_input, d_output, op_t{}, wrapped_init_t{}, input.size(), launch.get_stream());
  });
}

#ifdef TUNE_T
using uint_types = nvbench::type_list<TUNE_T>;
#else
#  if NVBENCH_HELPER_HAS_I128
using uint_types = nvbench::type_list<cuda::std::uint32_t, cuda::std::uint64_t, uint128_t>;
#  else
using uint_types = nvbench::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
#  endif
#endif

NVBENCH_BENCH_TYPES(inclusive_scan, NVBENCH_TYPE_AXES(uint_types, offset_types))
  .set_name("app-bicyclic-monoid")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
