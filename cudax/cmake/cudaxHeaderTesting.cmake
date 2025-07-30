# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Meta target for all configs' header builds:
add_custom_target(cudax.all.headers)

function(cudax_add_header_test label definitions)
  foreach(cn_target IN LISTS cudax_TARGETS)
    cudax_get_target_property(config_dialect ${cn_target} DIALECT)
    cudax_get_target_property(config_prefix ${cn_target} PREFIX)

    ###################
    # Non-STF headers #
    set(headertest_target ${config_prefix}.headers.${label}.no_stf)
    cccl_generate_header_tests(${headertest_target} cudax/include
      # The cudax header template removes the check for the `small` macro.
      HEADER_TEMPLATE "${cudax_SOURCE_DIR}/cmake/header_test.in.cu"
      GLOBS "cuda/experimental/*.cuh"
      EXCLUDES
        # The following internal headers are not required to compile independently:
        "cuda/experimental/__execution/prologue.cuh"
        "cuda/experimental/__execution/epilogue.cuh"
        # STF headers are compiled separately:
        "cuda/experimental/stf.cuh"
        "cuda/experimental/__stf/*"
    )
    target_link_libraries(${headertest_target} PUBLIC ${cn_target})
    target_compile_definitions(${headertest_target} PRIVATE
      ${definitions}
      "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
    )
    cudax_clone_target_properties(${headertest_target} ${cn_target})

    add_dependencies(cudax.all.headers ${headertest_target})
    add_dependencies(${config_prefix}.all ${headertest_target})

    # FIXME: Enable MSVC
    if (cudax_ENABLE_CUDASTF AND
        NOT "MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
      ###############
      # STF headers #
      set(headertest_target ${config_prefix}.headers.${label}.stf)
      cccl_generate_header_tests(${headertest_target} cudax/include
        GLOBS
          "cuda/experimental/stf.cuh"
          "cuda/experimental/__stf/*.cuh"

        # FIXME: The cudax header template removes the check for the `small` macro.
        # cuda/experimental/__stf/utility/memory.cuh defines functions named `small`.
        # These should be renamed to avoid conflicts with windows system headers, and
        # the following line removed:
        HEADER_TEMPLATE "${cudax_SOURCE_DIR}/cmake/header_test.in.cu"
      )
      target_link_libraries(${headertest_target} PUBLIC ${cn_target})
      target_compile_options(${headertest_target} PRIVATE
        # Required by stf headers:
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
        # FIXME: We should be able to refactor away from needing this by
        # using _CCCL_HOST_DEVICE and friends + `::cuda::std` utilities where
        # necessary.
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
      )
      cudax_clone_target_properties(${headertest_target} ${cn_target})
    endif()

    add_dependencies(cudax.all.headers ${headertest_target})
    add_dependencies(${config_prefix}.all ${headertest_target})
  endforeach()
endfunction()

cudax_add_header_test(basic "")
