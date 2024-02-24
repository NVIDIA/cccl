#include <cub/device/device_segmented_reduce.cuh>

int main()
{
  using offset_t = float; // error
  // using offset_t = int; // ok
  float *d_in{}, *d_out{};
  offset_t* d_offsets{};
  std::size_t temp_storage_bytes{};
  std::uint8_t* d_temp_storage{};

  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1, cub::Min(), 0);

  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);

  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);

  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);

  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);

  // expected-error {{"Offset iterator type should be integral."}}
  cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out, 0, d_offsets, d_offsets + 1);
}
