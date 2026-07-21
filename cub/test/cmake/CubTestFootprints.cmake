# GPU memory footprint classes for CUB tests.
#
# These classes are used by cub/test/CMakeLists.txt when assigning CTest
# RESOURCE_GROUPS. The percentages are scheduling claims only; they do not
# enforce a memory limit at runtime.
#
# - LARGE (100%): tests that should run alone on a GPU.
# - MEDIUM (50%): tests that may share a GPU with one other medium test.
# - SMALL (default): all remaining tests.
#
# The large and medium lists are based on the memory-heavy test tags from
# NVIDIA/cccl#9650 plus measured nvidia-smi peaks from Colossus validation.
# Measurement notes, RTX 3070 8 GiB:
# - medium: median ~730 MiB, max ~3002 MiB
# - small: median ~182 MiB, p99 ~388 MiB, max ~604 MiB

set(
  CCCL_TEST_GPU_PERCENT_SMALL
  15
  CACHE STRING
  "GPU percentage claimed by untagged CUB tests."
)
set(
  CCCL_TEST_GPU_PERCENT_MEDIUM
  50
  CACHE STRING
  "GPU percentage claimed by medium-footprint CUB tests."
)
set(
  CCCL_TEST_GPU_PERCENT_LARGE
  100
  CACHE STRING
  "GPU percentage claimed by large-footprint CUB tests."
)
mark_as_advanced(
  CCCL_TEST_GPU_PERCENT_SMALL
  CCCL_TEST_GPU_PERCENT_MEDIUM
  CCCL_TEST_GPU_PERCENT_LARGE
)

# gersemi: off
set(_cub_test_footprint_large
  device.batch_copy
  device.find
  device.find_bound_sorted_values
  device.histogram
  device.memcpy_batched
  device.merge
  device.merge_sort
  device.radix_sort_keys
  device.radix_sort_pairs
  device.scan_by_key_large_offsets
  device.scan_large_offsets
  device.segmented_radix_sort_keys
  device.segmented_radix_sort_pairs
  device.segmented_sort_keys
  device.segmented_sort_pairs
  device.select_if
  device.transform
)

set(_cub_test_footprint_medium
  device.copy_batched
  device.merge_no_unroll
  device.merge_sort_vsmem
  device.partition_flagged
  device.partition_if
  device.run_length_encode
  device.run_length_encode_non_trivial_runs
  device.scan
  device.segmented_reduce_large_offsets
  device.select_unique
  device.three_way_partition
  device.topk_keys
  device.topk_pairs
)
# gersemi: on

# Sets out_var to the GPU percentage the given test should claim.
function(cub_test_gpu_percent out_var test_name)
  if ("${test_name}" IN_LIST _cub_test_footprint_large)
    set(${out_var} ${CCCL_TEST_GPU_PERCENT_LARGE} PARENT_SCOPE)
  elseif ("${test_name}" IN_LIST _cub_test_footprint_medium)
    set(${out_var} ${CCCL_TEST_GPU_PERCENT_MEDIUM} PARENT_SCOPE)
  else()
    set(${out_var} ${CCCL_TEST_GPU_PERCENT_SMALL} PARENT_SCOPE)
  endif()
endfunction()
