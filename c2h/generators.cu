// SPDX-FileCopyrightText: Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_copy.cuh>

#include <thrust/detail/config/device_system.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/version.h>

#include <cuda/iterator>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <c2h/bfloat16.cuh>
#include <c2h/custom_type.h>
#include <c2h/detail/checked_memory.cuh>
#include <c2h/detail/generators.cuh>
#include <c2h/device_policy.h>
#include <c2h/extended_types.h>
#include <c2h/half.cuh>
#include <c2h/vector.h>

#if C2H_HAS_CURAND
#  include <curand.h>
#else
#  include <thrust/random.h>
#endif

namespace c2h::detail
{
#if C2H_HAS_CURAND
void check_curand_status(curandStatus_t status, const char* message)
{
  if (status != CURAND_STATUS_SUCCESS)
  {
    throw std::runtime_error{message};
  }
}
#else // C2H_HAS_CURAND
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
#endif // C2H_HAS_CURAND

class generator_state_t
{
public:
  generator_state_t(int device, ::cudaStream_t stream)
      : m_device(device)
      , m_stream(stream)
      , m_thread_id(thread_id_for_stream(stream))
  {
#if C2H_HAS_CURAND
    check_curand_status(curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT), "failed to create cuRAND generator");
#endif
  }

  static void destroy(generator_state_t* state) noexcept;

  [[nodiscard]] bool try_acquire() noexcept
  {
    const std::lock_guard<std::mutex> lock{m_state_mutex};
    if (m_is_leased)
    {
      return false;
    }

    m_is_leased = true;
    return true;
  }

  void release() noexcept
  {
    // A per-thread stream sentinel denotes a different stream when a lease is destroyed on another host thread.
    bool event_record_failed = m_thread_id != thread_id_for_stream(m_stream);
    if (!event_record_failed)
    {
      try
      {
        const scoped_current_device device_scope{m_device};
        if (m_completion_event == nullptr)
        {
          ::cudaEvent_t completion_event{};
          event_record_failed = cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming) != cudaSuccess;
          if (!event_record_failed)
          {
            m_completion_event = completion_event;
          }
        }

        if (!event_record_failed)
        {
          event_record_failed = cudaEventRecord(m_completion_event, m_stream) != cudaSuccess;
        }
      }
      catch (...)
      {
        event_record_failed = true;
      }
    }

    m_completion_event_record_failed = event_record_failed;

    const std::lock_guard<std::mutex> lock{m_state_mutex};
    m_is_leased = false;
  }

  [[nodiscard]] bool matches(int device, ::cudaStream_t stream) const noexcept
  {
    return m_device == device && m_stream == stream && m_thread_id == thread_id_for_stream(stream);
  }

  [[nodiscard]] int device() const noexcept
  {
    return m_device;
  }

  [[nodiscard]] float* prepare_random_generator(::cuda::stream_ref stream, seed_t seed, std::size_t num_items)
  {
    synchronize_previous_work();

    resize_distribution(num_items);
    if (num_items == 0)
    {
      return nullptr;
    }

#if C2H_HAS_CURAND
    check_curand_status(curandSetPseudoRandomGeneratorSeed(m_gen, seed.get()), "failed to seed cuRAND generator");
#else // C2H_HAS_CURAND
    m_gen.seed(seed.get());
#endif // C2H_HAS_CURAND

    generate(stream);

    return thrust::raw_pointer_cast(m_distribution.data());
  }

  // re-fills the currently held distribution vector with new random values
  void generate(::cuda::stream_ref stream)
  {
#if C2H_HAS_CURAND
    check_curand_status(curandSetStream(m_gen, stream.get()), "failed to set cuRAND generator stream");
    check_curand_status(
      curandGenerateUniform(m_gen, thrust::raw_pointer_cast(m_distribution.data()), m_distribution.size()),
      "failed to generate cuRAND distribution");
#else
    thrust::tabulate(device_policy.on(stream.get()), m_distribution.begin(), m_distribution.end(), i_to_rnd_t{m_gen});
    m_gen.discard(m_distribution.size());
#endif
  }

private:
  ~generator_state_t()
  {
    cleanup_on_current_device();
  }

  void synchronize_previous_work()
  {
    if (m_completion_event == nullptr && !m_completion_event_record_failed)
    {
      return;
    }

    const scoped_current_device device_scope{m_device};
    const cudaError_t status =
      m_completion_event_record_failed ? cudaDeviceSynchronize() : cudaEventSynchronize(m_completion_event);
    if (status != cudaSuccess)
    {
      throw ::cuda::cuda_error{status, "failed to synchronize random generator state"};
    }

    m_completion_event_record_failed = false;
  }

  void cleanup_on_current_device() noexcept
  {
    if (m_completion_event_record_failed)
    {
      // Destructors cannot report synchronization failures.
      (void) cudaDeviceSynchronize();
    }
    else if (m_completion_event != nullptr && cudaEventSynchronize(m_completion_event) != cudaSuccess)
    {
      // Fall back to synchronizing the device before releasing the distribution storage.
      (void) cudaDeviceSynchronize();
    }

    // Release the distribution storage before the owning-device scope exits.
    {
      c2h::device_vector<float> distribution_to_release;
      distribution_to_release.swap(m_distribution);
    }

    if (m_completion_event != nullptr)
    {
      // Destructors cannot report event cleanup failures.
      (void) cudaEventDestroy(m_completion_event);
      m_completion_event = nullptr;
    }

#if C2H_HAS_CURAND
    // Destructors cannot report generator cleanup failures.
    (void) curandDestroyGenerator(m_gen);
#endif
  }

  [[nodiscard]] static std::thread::id thread_id_for_stream(::cudaStream_t stream) noexcept
  {
    if (stream == cudaStreamPerThread)
    {
      return std::this_thread::get_id();
    }

#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
    if (stream == ::cudaStream_t{})
    {
      return std::this_thread::get_id();
    }
#endif // CUDA_API_PER_THREAD_DEFAULT_STREAM

    return {};
  }

  void resize_distribution(std::size_t num_items)
  {
#if THRUST_VERSION >= 300100
    m_distribution.resize(num_items, thrust::no_init);
#else // THRUST_VERSION >= 300100
    m_distribution.resize(num_items);
#endif // THRUST_VERSION >= 300100
  }

#if C2H_HAS_CURAND
  curandGenerator_t
#else
  thrust::default_random_engine
#endif
    m_gen;
  c2h::device_vector<float> m_distribution;
  int m_device;
  ::cudaStream_t m_stream;
  std::thread::id m_thread_id;
  std::mutex m_state_mutex;
  ::cudaEvent_t m_completion_event      = nullptr;
  bool m_completion_event_record_failed = false;
  bool m_is_leased                      = false;
};

void generator_state_t::destroy(generator_state_t* state) noexcept
{
  try
  {
    const scoped_current_device device_scope{state->device()};
    delete state;
  }
  catch (...)
  {
    // CUDA resources cannot be safely destroyed unless their owning device is current.
    // Retain the state and let process teardown reclaim it if device selection fails.
  }
}

random_data_t::random_data_t(float* data, std::shared_ptr<generator_state_t> state) noexcept
    : m_data(data)
    , m_state(std::move(state))
{}

random_data_t::random_data_t(random_data_t&& other) noexcept
    : m_data(std::exchange(other.m_data, nullptr))
    , m_state(std::move(other.m_state))
{}

random_data_t& random_data_t::operator=(random_data_t&& other) noexcept
{
  if (this != &other)
  {
    if (m_state)
    {
      m_state->release();
    }

    m_data  = std::exchange(other.m_data, nullptr);
    m_state = std::move(other.m_state);
  }
  return *this;
}

random_data_t::~random_data_t()
{
  if (m_state)
  {
    m_state->release();
  }
}

class generator_t
{
public:
  // Explicit bodies prevent nvcc from inferring host/device special members for this host-only state.
  generator_t() {} // NOLINT(modernize-use-equals-default)
  ~generator_t() {} // NOLINT(modernize-use-equals-default)

  [[nodiscard]] random_data_t prepare_random_generator(seed_t seed, std::size_t num_items)
  {
    int device{};
    const cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess)
    {
      throw ::cuda::cuda_error{status, "failed to get current device"};
    }

    const ::cuda::stream_ref stream{::cudaStream_t{}};
    return prepare_random_generator(device, stream, seed, num_items);
  }

  [[nodiscard]] random_data_t prepare_random_generator(::cuda::stream_ref stream, seed_t seed, std::size_t num_items)
  {
    if (stream.get() == ::cudaStream_t{})
    {
      return prepare_random_generator(seed, num_items);
    }

    const int device = stream.device().get();
    const scoped_current_device device_scope{device};
    return prepare_random_generator(device, stream, seed, num_items);
  }

  [[nodiscard]] std::size_t cached_state_count()
  {
    const std::lock_guard<std::mutex> lock{m_states_mutex};
    return m_states.size();
  }

private:
  [[nodiscard]] random_data_t
  prepare_random_generator(int device, ::cuda::stream_ref stream, seed_t seed, std::size_t num_items)
  {
    auto state = state_for(device, stream.get());
    try
    {
      float* data = state->prepare_random_generator(stream, seed, num_items);
      return random_data_t{data, std::move(state)};
    }
    catch (...)
    {
      state->release();
      throw;
    }
  }

  [[nodiscard]] std::shared_ptr<generator_state_t> state_for(int device, ::cudaStream_t stream)
  {
    std::shared_ptr<generator_state_t> result;
    std::shared_ptr<generator_state_t> evicted;

    {
      const std::lock_guard<std::mutex> lock{m_states_mutex};
      for (auto state = m_states.begin(); state != m_states.end(); ++state)
      {
        if ((*state)->matches(device, stream) && (*state)->try_acquire())
        {
          result = *state;
          m_states.erase(state);
          m_states.push_back(result);
          break;
        }
      }

      if (!result)
      {
        result = std::shared_ptr<generator_state_t>{new generator_state_t{device, stream}, &generator_state_t::destroy};
        [[maybe_unused]] const bool acquired = result->try_acquire();
        _CCCL_VERIFY(acquired, "");

        if (m_states.size() >= max_cached_generator_states)
        {
          evicted = std::move(m_states.front());
          m_states.erase(m_states.begin());
        }
        m_states.push_back(result);
      }
    }

    // An in-use state is kept alive by its caller. An idle state waits for its last consumer before releasing storage.
    evicted.reset();
    return result;
  }

  std::vector<std::shared_ptr<generator_state_t>> m_states;
  std::mutex m_states_mutex;
};

// global generator state
cuda::std::optional<generator_t> generator;

void init_generator()
{
  _CCCL_VERIFY(!generator.has_value(), "");
  generator.emplace();
}

random_data_t prepare_random_data(seed_t seed, std::size_t num_items)
{
  return generator.value().prepare_random_generator(seed, num_items);
}

random_data_t prepare_random_data(::cuda::stream_ref stream, seed_t seed, std::size_t num_items)
{
  return generator.value().prepare_random_generator(stream, seed, num_items);
}

std::size_t cached_generator_state_count()
{
  return generator.value().cached_state_count();
}

void cleanup_generator()
{
  _CCCL_VERIFY(generator.has_value(), "");
  generator.reset();
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
  custom_type_state_t min,
  custom_type_state_t max,
  std::size_t elements,
  std::size_t element_size)
{
  gen_custom_type_state(::cuda::stream_ref{::cudaStream_t{}}, seed, d_out, min, max, elements, element_size);
}

void gen_custom_type_state(
  ::cuda::stream_ref stream,
  seed_t seed,
  char* d_out,
  custom_type_state_t /* min */,
  custom_type_state_t /* max */,
  std::size_t elements,
  std::size_t element_size)
{
  // FIXME(bgruber): implement min/max handling for custom_type_state_t
  const auto random_data = prepare_random_data(stream, seed, elements * 2);
  thrust::for_each(device_policy.on(stream.get()),
                   thrust::counting_iterator<std::size_t>{0},
                   thrust::counting_iterator<std::size_t>{elements},
                   random_to_custom_t{random_data.data(), d_out, element_size});
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
  __host__ __device__ __forceinline__ cuda::constant_iterator<T> operator()(std::size_t i)
  {
    return cuda::constant_iterator<T>(static_cast<T>(i));
  }
};

template <>
struct repeat_index_t<custom_type_state_t>
{
  __host__ __device__ __forceinline__ cuda::constant_iterator<custom_type_state_t> operator()(std::size_t i)
  {
    custom_type_state_t item{};
    item.key = i;
    item.val = i;
    return cuda::constant_iterator<custom_type_state_t>(item);
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

#  if THRUST_VERSION >= 300100
  device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
#  else
  device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
#  endif // THRUST_VERSION >= 300100

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
