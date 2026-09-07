# Configures a target for the STF framework.
function(cudax_stf_configure_target target_name)
  set(options LINK_MATHLIBS)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(
    CSCT
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  target_link_libraries(
    ${target_name}
    PRIVATE #
      CUDA::cudart_static
      CUDA::cuda_driver
  )

  # STF itself does not use cuRAND; only a couple of examples/tests do (they are dropped
  # from their source lists when the component is missing). Link it when the toolkit has
  # it, but do not make every STF target depend on a partial install having it.
  if (TARGET CUDA::curand)
    target_link_libraries(${target_name} PRIVATE CUDA::curand)
  endif()

  if (cudax_ENABLE_CUDASTF_CODE_GENERATION)
    target_compile_options(
      ${target_name}
      PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
    )
  else()
    target_compile_definitions(
      ${target_name}
      PRIVATE "CUDASTF_DISABLE_CODE_GENERATION"
    )
  endif()

  target_compile_options(
    ${target_name}
    PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
  )

  set_target_properties(
    ${target_name}
    PROPERTIES #
      CUDA_RUNTIME_LIBRARY Static
      CUDA_SEPARABLE_COMPILATION ON
  )

  if (CSCT_LINK_MATHLIBS)
    target_link_libraries(
      ${target_name}
      PRIVATE #
        CUDA::cublas
        CUDA::cusolver
    )
  endif()

  if (cudax_ENABLE_CUDASTF_BOUNDSCHECK)
    target_compile_definitions(${target_name} PRIVATE "CUDASTF_BOUNDSCHECK")
  endif()
endfunction()
