// SPDX-FileCopyrightText: Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_copy.cuh>

#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tabulate.h>

#include <c2h/bfloat16.cuh>
#include <c2h/custom_type.h>
#include <c2h/detail/generators.cuh>
#include <c2h/device_policy.h>
#include <c2h/extended_types.h>
#include <c2h/generators.h>
#include <c2h/half.cuh>
#include <c2h/vector.h>

#if C2H_HAS_CURAND
#  include <curand.h>
#else
#  include <thrust/random.h>
#endif

namespace c2h::detail
{

#if !C2H_HAS_CURAND
struct i_to_rnd_t
{
  __host__ __device__ i_to_rnd_t(thrust::default_random_engine engine)
      : m_engine(engine)
  {}

  thrust::default_random_engine m_engine{};

  template <typename IndexType>
  __host__ __device__ float operator()(IndexType n)
  {
    m_engine.discard(n);
    return thrust::uniform_real_distribution<float>{0.0f, 1.0f}(m_engine);
  }
};
#endif // !C2H_HAS_CURAND

void generator_t::generate()
{
#if C2H_HAS_CURAND
  curandGenerateUniform(m_gen, thrust::raw_pointer_cast(m_distribution.data()), m_distribution.size());
#else
  thrust::tabulate(device_policy, m_distribution.begin(), m_distribution.end(), i_to_rnd_t{m_re});
  m_re.discard(m_distribution.size());
#endif
}

float* generator_t::prepare_random_generator(seed_t seed, std::size_t num_items)
{
  m_distribution.resize(num_items);

#if C2H_HAS_CURAND
  curandSetPseudoRandomGeneratorSeed(m_gen, seed.get());
#else
  m_re.seed(seed.get());
#endif

  generate();

  return thrust::raw_pointer_cast(m_distribution.data());
}

struct random_to_custom_t
{
  static constexpr std::size_t m_max_key = std::numeric_limits<std::size_t>::max();

  __device__ void operator()(std::size_t idx) const
  {
    auto out = reinterpret_cast<custom_type_state_t*>(m_out + idx * m_element_size);
    out->key = static_cast<std::size_t>(static_cast<float>(m_max_key) * m_in[idx * 2 + 0]);
    out->val = static_cast<std::size_t>(static_cast<float>(m_max_key) * m_in[idx * 2 + 1]);
  }

  float* m_in{};
  char* m_out{};
  std::size_t m_element_size{};
};

void gen_custom_type_state(
  seed_t seed,
  char* d_out,
  custom_type_state_t /* min */,
  custom_type_state_t /* max */,
  std::size_t elements,
  std::size_t element_size)
{
  // FIXME(bgruber): implement min/max handling for custom_type_state_t
  float* d_in = generator.prepare_random_generator(seed, elements * 2);
  thrust::for_each(device_policy,
                   thrust::counting_iterator<std::size_t>{0},
                   thrust::counting_iterator<std::size_t>{elements},
                   random_to_custom_t{d_in, d_out, element_size});
}

template <typename T>
struct spaced_out_it_op
{
  char* base_it;
  std::size_t element_size;

  __host__ __device__ __forceinline__ T& operator()(std::size_t offset) const
  {
    return *reinterpret_cast<T*>(base_it + (element_size * offset));
  }
};

template <typename T>
struct offset_to_iterator_t
{
  char* base_it;
  std::size_t element_size;

  __host__
    __device__ __forceinline__ thrust::transform_iterator<spaced_out_it_op<T>, thrust::counting_iterator<std::size_t>>
    operator()(std::size_t offset) const
  {
    // The pointer to the beginning of this "buffer" (aka a series of same "keys")
    auto base_ptr = base_it + (element_size * offset);

    // We need to make sure that the i-th element within this "buffer" is spaced out by
    // `element_size`
    auto counting_it = thrust::make_counting_iterator(std::size_t{0});
    spaced_out_it_op<T> space_out_op{base_ptr, element_size};
    return thrust::make_transform_iterator(counting_it, space_out_op);
  }
};

template <class T>
struct repeat_index_t
{
  __host__ __device__ __forceinline__ thrust::constant_iterator<T> operator()(std::size_t i)
  {
    return thrust::constant_iterator<T>(static_cast<T>(i));
  }
};

template <>
struct repeat_index_t<custom_type_state_t>
{
  __host__ __device__ __forceinline__ thrust::constant_iterator<custom_type_state_t> operator()(std::size_t i)
  {
    custom_type_state_t item{};
    item.key = i;
    item.val = i;
    return thrust::constant_iterator<custom_type_state_t>(item);
  }
};

template <typename OffsetT>
struct offset_to_size_t
{
  const OffsetT* offsets;

  __host__ __device__ __forceinline__ std::size_t operator()(std::size_t i)
  {
    return offsets[i + 1] - offsets[i];
  }
};

/**
 * @brief Initializes key-segment ranges from an offsets-array like the one given by
 * `gen_uniform_offset`.
 */
template <typename OffsetT, typename KeyT>
void init_key_segments(::cuda::std::span<const OffsetT> segment_offsets, KeyT* d_out, std::size_t element_size)
{
  OffsetT total_segments   = static_cast<OffsetT>(segment_offsets.size() - 1);
  const OffsetT* d_offsets = segment_offsets.data();

  thrust::counting_iterator<int> iota(0);
  offset_to_iterator_t<KeyT> dst_transform_op{reinterpret_cast<char*>(d_out), element_size};

  auto d_range_srcs  = thrust::make_transform_iterator(iota, repeat_index_t<KeyT>{});
  auto d_range_dsts  = thrust::make_transform_iterator(d_offsets, dst_transform_op);
  auto d_range_sizes = thrust::make_transform_iterator(iota, offset_to_size_t<OffsetT>{d_offsets});

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  std::uint8_t* d_temp_storage   = nullptr;
  std::size_t temp_storage_bytes = 0;
  // TODO(bgruber): replace by a non-CUB implementation
  cub::DeviceCopy::Batched(
    d_temp_storage, temp_storage_bytes, d_range_srcs, d_range_dsts, d_range_sizes, total_segments);

  device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // TODO(bgruber): replace by a non-CUB implementation
  cub::DeviceCopy::Batched(
    d_temp_storage, temp_storage_bytes, d_range_srcs, d_range_dsts, d_range_sizes, total_segments);
  cudaDeviceSynchronize();
#else // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  static_assert(sizeof(OffsetT) == 0, "Need to implement a non-CUB version of cub::DeviceCopy::Batched");
  // TODO(bgruber): implement and *test* a non-CUB version, here is a sketch:
  // thrust::for_each(
  //   thrust::device,
  //   thrust::counting_iterator<OffsetT>{0},
  //   thrust::counting_iterator<OffsetT>{total_segments},
  //   [&](OffsetT i) {
  //     const auto value = d_range_srcs[i];
  //     const auto start = d_range_sizes[i];
  //     const auto end   = d_range_sizes[i + 1];
  //     for (auto j = start; j < end; ++j)
  //     {
  //       d_range_dsts[j] = value;
  //     }
  //   });
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
}

template void
init_key_segments(::cuda::std::span<const std::uint32_t> segment_offsets, std::int32_t* out, std::size_t element_size);
template void
init_key_segments(::cuda::std::span<const std::uint32_t> segment_offsets, std::uint8_t* out, std::size_t element_size);
template void
init_key_segments(::cuda::std::span<const std::uint32_t> segment_offsets, float* out, std::size_t element_size);
template void init_key_segments(
  ::cuda::std::span<const std::uint32_t> segment_offsets, custom_type_state_t* out, std::size_t element_size);
#if TEST_HALF_T()
template void
init_key_segments(::cuda::std::span<const std::uint32_t> segment_offsets, half_t* out, std::size_t element_size);
#endif // TEST_HALF_T()

#if TEST_BF_T()
template void
init_key_segments(::cuda::std::span<const std::uint32_t> segment_offsets, bfloat16_t* out, std::size_t element_size);
#endif // TEST_BF_T()

} // namespace c2h::detail
