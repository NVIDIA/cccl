## thrust_add_example
#
# Add an example executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the example
#   target. Useful for post-processing target information per-backend.
# example_name: The name of the example minus "<config_prefix>.example." For
#   instance, examples/vector.cu will be "vector", and examples/cuda/copy.cu
#   would be "cuda.copy".
# example_src: The source file that implements the example.
# thrust_target: The reference thrust target with configuration information.
#
function(
  thrust_add_example
  target_name_var
  example_name
  example_src
  thrust_target
)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  # Wrap the .cu file in .cpp for non-CUDA backends
  if ("CUDA" STREQUAL "${config_device}")
    set(real_example_src "${example_src}")
  else()
    thrust_wrap_cu_in_cpp(real_example_src "${example_src}" ${thrust_target})
  endif()

  # The actual name of the test's target:
  set(example_target ${config_prefix}.example.${example_name})
  set(${target_name_var} ${example_target} PARENT_SCOPE)

  cccl_add_executable(${example_target} SOURCES "${real_example_src}" ADD_CTEST)
  target_link_libraries(${example_target} PRIVATE ${thrust_target})
  target_include_directories(
    ${example_target}
    PRIVATE "${Thrust_SOURCE_DIR}/examples"
  )

  if ("CUDA" STREQUAL "${config_device}")
    cccl_set_rdc(${example_target} ${THRUST_FORCE_RDC})
  endif()

  # We do not want to explicitly include `host_device.h` if not needed,
  # so force include the file for non CUDA targets.
  set(gx_msvc "$<CXX_COMPILER_ID:MSVC>")
  set(gx_cxx "$<COMPILE_LANGUAGE:CXX>")
  set(gx_cxx_msvc "$<AND:${gx_cxx},${gx_msvc}>")
  set(gx_cxx_not_msvc "$<AND:${gx_cxx},$<NOT:${gx_msvc}>>")
  target_compile_options(
    ${example_target}
    PRIVATE
      "$<${gx_cxx_msvc}:SHELL:/FI include/host_device.h>"
      "$<${gx_cxx_not_msvc}:SHELL:-include include/host_device.h>"
  )
  target_compile_definitions(
    ${example_target}
    PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:THRUST_EXAMPLE_DEVICE_SIDE>
  )

  # Run OMP/TBB tests in serial. Multiple OMP processes will massively
  # oversubscribe the machine with GCC's OMP, and we want to test these with
  # the full CPU available to each unit test.
  set(config_systems ${config_host} ${config_device})
  if (("OMP" IN_LIST config_systems) OR ("TBB" IN_LIST config_systems))
    set_tests_properties(${example_target} PROPERTIES RUN_SERIAL ON)
  endif()

  # Check for per-example script. Script will be included in the current scope
  # to allow custom property modifications.
  get_filename_component(example_cmake_script "${example_src}" NAME_WLE)
  set(
    example_cmake_script
    "${CMAKE_CURRENT_LIST_DIR}/${example_cmake_script}.cmake"
  )
  # Use a glob so we can detect if this changes:
  file(
    GLOB example_cmake_script
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
    CONFIGURE_DEPENDS
    "${example_cmake_script}"
  )
  if (example_cmake_script) # Will be non-empty only if the script exists
    include("${example_cmake_script}")
  endif()
endfunction()

file(
  GLOB example_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cu
  *.cpp
)

foreach (thrust_target IN LISTS THRUST_TARGETS)
  foreach (example_src IN LISTS example_srcs)
    get_filename_component(example_name "${example_src}" NAME_WLE)
    thrust_add_example(example_target ${example_name} "${example_src}" ${thrust_target})
  endforeach()
endforeach()

add_subdirectory(cuda)
