# Configures a target for the Places framework.
function(cudax_places_configure_target target_name)
  target_link_libraries(
    ${target_name}
    PRIVATE #
      CUDA::cudart_static
      CUDA::cuda_driver
  )

  target_compile_options(
    ${target_name}
    PRIVATE
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
  )

  set_target_properties(
    ${target_name}
    PROPERTIES #
      CUDA_RUNTIME_LIBRARY Static
      CUDA_SEPARABLE_COMPILATION ON
  )
endfunction()
