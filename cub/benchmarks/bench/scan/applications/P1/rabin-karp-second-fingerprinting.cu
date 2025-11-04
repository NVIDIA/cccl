// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_scan.cuh>

#include <thrust/host_vector.h>

#include <cuda/std/limits>

#include <iostream>

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
/* Denote epsilon, the identity element, be an empty sequence, and consider
 * set of sequences of {0, 1} bits, with binary operation of concatenation.
 *
 * Define homomorphism K from the set of sequence to 2-by-2 integral matrices
 * over cyclic ring Z_p for some prime p.
 *
 *    K( ''  ) = [[ 1, 0], [0, 1]]
 *    K( '0' ) = [[1, 0], [1, 1]]
 *    K( '1' ) = [[1, 1], [0, 1]]
 *
 *    K( concat(seq1, seq2) ) := matmul( K(seq1), K(seq2) ) in Z_p
 *
 * Given a sequence of unsigned integers, encoding bit sequences,
 * we build transform iterator mapping integer to the matrix. Then
 * call inclusive_scan with matrix multiply operator in Z_p
 *
 * Ref: https://doi.org/10.1147/rd.312.0249
 */

// Types associated with the cyclic ring
using ZpT   = cuda::std::uint32_t;
using WideT = cuda::std::uint64_t;

using MatT = cuda::std::array<ZpT, 4>;

inline ZpT __host__ __device__ Zp_mul(ZpT v1, ZpT v2, cuda::fast_mod_div<WideT> m_p)
{
  const auto w1 = static_cast<WideT>(v1);
  const auto w2 = static_cast<WideT>(v2);
  return static_cast<ZpT>((w1 * w2) % m_p);
}

inline ZpT __host__ __device__ Zp_add(ZpT v1, ZpT v2, cuda::fast_mod_div<WideT> m_p)
{
  const auto w1 = static_cast<WideT>(v1);
  const auto w2 = static_cast<WideT>(v2);
  return static_cast<ZpT>((w1 + w2) % m_p);
}

inline MatT __host__ __device__ Zp_matmul(MatT v1, MatT v2, cuda::fast_mod_div<WideT> m_p)
{
  ZpT _1_00_2_00 = Zp_mul(v1[0], v2[0], m_p);
  ZpT _1_01_2_10 = Zp_mul(v1[1], v2[2], m_p);
  ZpT _r_00      = Zp_add(_1_00_2_00, _1_01_2_10, m_p);

  ZpT _1_00_2_01 = Zp_mul(v1[0], v2[1], m_p);
  ZpT _1_01_2_11 = Zp_mul(v1[1], v2[3], m_p);
  ZpT _r_01      = Zp_add(_1_00_2_01, _1_01_2_11, m_p);

  ZpT _1_10_2_00 = Zp_mul(v1[2], v2[0], m_p);
  ZpT _1_11_2_10 = Zp_mul(v1[3], v2[2], m_p);
  ZpT _r_10      = Zp_add(_1_10_2_00, _1_11_2_10, m_p);

  ZpT _1_10_2_01 = Zp_mul(v1[2], v2[1], m_p);
  ZpT _1_11_2_11 = Zp_mul(v1[3], v2[3], m_p);
  ZpT _r_11      = Zp_add(_1_10_2_01, _1_11_2_11, m_p);

  return {_r_00, _r_01, _r_10, _r_11};
}

struct RabinKarpOp
{
  cuda::fast_mod_div<WideT> m_p;

  __host__ __device__ RabinKarpOp(ZpT p)
      : m_p(static_cast<WideT>(p))
  {}

  // scan operator: non-commutative and associative
  MatT __host__ __device__ operator()(MatT v1, MatT v2) const
  {
    return Zp_matmul(v1, v2, m_p);
  }
};

template <typename T>
struct ChunkToMat
{
  static_assert(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>, "Bit sequence should be represented");

  cuda::fast_mod_div<WideT> m_p;

  __host__ __device__ ChunkToMat(ZpT p)
      : m_p(static_cast<WideT>(p))
  {}

  MatT __host__ __device__ operator()(const T& bits) const
  {
    static constexpr int n_bits = cuda::std::numeric_limits<T>::digits;
    static_assert(n_bits >= 1, "Type must have non-zero bitwidth");

    static constexpr MatT _0 = {ZpT{1}, ZpT{0}, ZpT{1}, ZpT{1}}; // [[1, 0], [1, 1]]
    static constexpr MatT _1 = {ZpT{1}, ZpT{1}, ZpT{0}, ZpT{1}}; // [[1, 1], [0, 1]]

    // initialize with identity matrix
    MatT m  = (bits & 1) ? _1 : _0;
    T _bits = bits >> 1;

    // use of cuda::static_for here results in performance regression due to increased register pressure
    for (int i = 1; i < n_bits; ++i)
    {
      (void) i;
      m = Zp_matmul((_bits & 1) ? _1 : _0, m, m_p);
      _bits >>= 1;
    }

    return m;
  }
};

// Iterator that performs assignment at specific index only, discards otherwise
//
// This iterator allows tp use inclusive_scan to perform reduction with
// non-commutative associative binary operator
//
template <typename OffsetT, typename Iter>
struct write_at_specific_index_or_discard
{
private:
  OffsetT m_index{};
  OffsetT m_target_index;
  Iter m_iter;

  void __host__ __device__ set_index(OffsetT index)
  {
    m_index = index;
  }

public:
  struct assign_proxy
  {
  private:
    bool m_writable;
    Iter m_iter;

  public:
    __host__ __device__ assign_proxy(bool writable, Iter iter)
        : m_writable(writable)
        , m_iter(iter)
    {}

    template <typename Tp>
    constexpr assign_proxy& __host__ __device__ operator=(Tp&& v)
    {
      if (m_writable)
      {
        *m_iter = v;
      }
      return *this;
    }
  };

  using iterator_concept  = cuda::std::random_access_iterator_tag;
  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type        = cuda::std::iter_value_t<Iter>;
  using difference_type   = cuda::std::iter_difference_t<Iter>;
  using pointer           = void;
  using reference         = void;

  write_at_specific_index_or_discard() = delete;
  explicit __host__ __device__ write_at_specific_index_or_discard(OffsetT offset, Iter iter)
      : m_target_index(offset)
      , m_iter(iter)
  {}

  write_at_specific_index_or_discard(const write_at_specific_index_or_discard&)            = default;
  write_at_specific_index_or_discard(write_at_specific_index_or_discard&&)                 = default;
  write_at_specific_index_or_discard& operator=(const write_at_specific_index_or_discard&) = default;
  write_at_specific_index_or_discard& operator=(write_at_specific_index_or_discard&&)      = default;

  assign_proxy __host__ __device__ operator[](difference_type n)
  {
    return {(m_index + static_cast<OffsetT>(n)) == m_target_index, m_iter};
  }

  write_at_specific_index_or_discard __host__ __device__ operator+(difference_type n) const
  {
    auto r = write_at_specific_index_or_discard(m_target_index, m_iter);
    r.set_index((m_index + static_cast<OffsetT>(n)));
    return r;
  }
};

template <typename InputT, typename OutputT>
[[nodiscard]] bool validate(
  const thrust::device_vector<InputT>& input, const thrust::device_vector<OutputT>& output, ZpT p, cudaStream_t stream)
{
  using accum_t = OutputT;
  using input_t = InputT;

  cudaStreamSynchronize(stream);

  thrust::host_vector<accum_t> h_out(output);
  thrust::host_vector<input_t> h_inp(input);

  accum_t ref_mat = {1, 0, 0, 1};

  static constexpr accum_t mat_0 = {1, 0, 1, 1}; // lower diagonal
  static constexpr accum_t mat_1 = {1, 1, 0, 1}; // upper diagonal
  cuda::fast_mod_div<impl::WideT> mod(p);
  for (auto&& el : h_inp)
  {
    input_t v = el;

    accum_t word_mat = {1, 0, 0, 1};
    for (int i = 0; i < sizeof(input_t) * 8; ++i)
    {
      if (v & 1)
      {
        word_mat = impl::Zp_matmul(mat_1, word_mat, mod);
      }
      else
      {
        word_mat = impl::Zp_matmul(mat_0, word_mat, mod);
      }
      v >>= 1;
    }

    ref_mat = impl::Zp_matmul(ref_mat, word_mat, mod);
  }

  const accum_t& res = h_out[0];
  if (ref_mat != res)
  {
    std::cout << "FAILED: ";
    std::cout << "cub_computed([[" << res[0] << ", " << res[1] << "], [" << res[2] << ", " << res[3] << "]]) != ";
    std::cout
      << "reference([[" << ref_mat[0] << ", " << ref_mat[1] << "], [" << ref_mat[2] << ", " << ref_mat[3] << "]])\n";
    return false;
  }

  return true;
}
}; // namespace impl

template <typename BitsetT, typename OffsetT>
static void inclusive_scan(nvbench::state& state, nvbench::type_list<BitsetT, OffsetT>)
{
  using wrapped_init_t = cub::NullType;
  using op_t           = impl::RabinKarpOp;
  using input_t        = BitsetT;
  using raw_it_t       = const input_t*;
  using input_it_t     = cuda::transform_iterator<impl::ChunkToMat<input_t>, raw_it_t>;
  using accum_t        = impl::MatT;
  using output_ptr_t   = impl::MatT*;
  using output_it_t    = impl::write_at_specific_index_or_discard<OffsetT, output_ptr_t>;
  using offset_t       = cub::detail::choose_offset_t<OffsetT>;

  using ZpT = impl::ZpT;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<accum_t>;
  using dispatch_t = cub::
    DispatchScan<input_it_t, output_it_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No, policy_t>;
#else
  using dispatch_t =
    cub::DispatchScan<input_it_t, output_it_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No>;
#endif

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<input_t> input = generate(elements);
  thrust::device_vector<accum_t> output(1, thrust::no_init);

  // a large prime
  ZpT p = static_cast<ZpT>(state.get_int64("Modulus"));

  raw_it_t d_input      = thrust::raw_pointer_cast(input.data());
  output_ptr_t d_output = thrust::raw_pointer_cast(output.data());

  input_it_t inp_it(d_input, impl::ChunkToMat<input_t>(p));
  output_it_t out_it(static_cast<OffsetT>(elements - 1), d_output);

  state.add_element_count(elements);
  state.add_global_memory_reads<input_t>(elements, "Sequence Size");
  state.add_global_memory_writes<accum_t>(1, "Hash Size");

  cudaStream_t bench_stream = state.get_cuda_stream().get_stream();

  size_t tmp_size;
  dispatch_t::Dispatch(nullptr, tmp_size, inp_it, out_it, op_t{p}, wrapped_init_t{}, input.size(), bench_stream);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);
  nvbench::uint8_t* d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      d_tmp,
      tmp_size,
      inp_it,
      out_it, // iterator that only writes the last element of inclusive prefix scan sequence,
      op_t{p},
      wrapped_init_t{},
      input.size(),
      launch.get_stream());
  });

  // for validation uncomment these two lines
  // assert(impl::validate(input, output, p, bench_stream));
}

#ifdef TUNE_T
using type_list = nvbench::type_list<TUNE_T>;
#else
// we can split stream of bits into 8-bit, 16-bit, etc. chunks, effectively
// serving as the number of bits processed by a thread
using type_list = nvbench::type_list<cuda::std::uint8_t, cuda::std::uint16_t, cuda::std::uint32_t, cuda::std::uint64_t>;
#endif

NVBENCH_BENCH_TYPES(inclusive_scan, NVBENCH_TYPE_AXES(type_list, offset_types))
  .set_name("rabin-karp-fingerprinting-monoid")
  .set_type_axes_names({"BitsetT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_axis("Modulus", {2725841});
