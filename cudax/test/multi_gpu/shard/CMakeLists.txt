string(APPEND TEST_PREFIX ".shard")

set(test_src shard.cu)

foreach (src IN LISTS test_src)
  cmake_path(GET src STEM stem)
  cudax_add_catch2_test(test_target "${TEST_PREFIX}.${stem}" "${src}")
endforeach()
