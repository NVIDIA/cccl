# Configures a target for the STF framework.
function(cudax_stf_configure_target target_name)
  set(options LINK_MATHLIBS)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(CSCT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  target_link_libraries(${target_name} PRIVATE
    ${cn_target}
    CUDA::cudart
    CUDA::curand
    CUDA::cuda_driver
  )

  if (cudax_ENABLE_CUDASTF_CODE_GENERATION)
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
    )
  else()
    target_compile_definitions(${target_name} PRIVATE "CUDASTF_DISABLE_CODE_GENERATION")
  endif()

  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
  )

  set_target_properties(${target_name} PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION ON
  )

  if (CSCT_LINK_MATHLIBS)
    target_link_libraries(${target_name} PRIVATE
      CUDA::cublas
      CUDA::cusolver
    )
  endif()

  if (cudax_ENABLE_CUDASTF_BOUNDSCHECK)
    target_compile_definitions(${target_name} PRIVATE
      "CUDASTF_BOUNDSCHECK"
    )
  endif()
endfunction()
