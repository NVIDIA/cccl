# We need to test cuda::proclaim_return_type
if ("NVIDIA" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    target_compile_options(${test_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)
endif()
