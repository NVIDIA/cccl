#include <cub/device/device_radix_sort.cuh>

struct custom_t
{
  std::uint16_t i;
  float f;
};

struct decomposer_t
{
  // expected-error {{"DecomposerT must be a callable object returning a tuple of references"}}
  __host__ __device__ std::uint16_t& operator()(custom_t& key) const
  {
    return key.i;
  }
};

int main()
{
  custom_t *d_in{};
  custom_t *d_out{};
  std::size_t temp_storage_bytes{};
  std::uint8_t *d_temp_storage{};

  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, decomposer_t{});
}
