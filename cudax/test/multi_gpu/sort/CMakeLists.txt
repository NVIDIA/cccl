string(APPEND TEST_PREFIX ".sort")

cudax_add_catch2_test(test_target "${TEST_PREFIX}.sort" sort.cu)
target_link_libraries(${test_target} PRIVATE NCCL::NCCL)
