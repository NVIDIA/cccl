# Given a cu_file (e.g. foo/bar.cu) relative to CMAKE_CURRENT_SOURCE_DIR
# and a thrust_target, create a cpp file that includes the .cu file, and set
# ${cpp_file_var} in the parent scope to the full path of the new file. The new
# file will be generated in:
# ${CMAKE_CURRENT_BINARY_DIR}/<thrust_target_prefix>/${cu_file}.cpp
function(thrust_wrap_cu_in_cpp cpp_file_var cu_file thrust_target)
  thrust_get_target_property(prefix ${thrust_target} PREFIX)
  set(wrapped_source_file "${CMAKE_CURRENT_SOURCE_DIR}/${cu_file}")
  set(cpp_file "${CMAKE_CURRENT_BINARY_DIR}/${prefix}/${cu_file}.cpp")
  configure_file(
    "${Thrust_SOURCE_DIR}/cmake/wrap_source_file.cpp.in"
    "${cpp_file}"
  )
  set(${cpp_file_var} "${cpp_file}" PARENT_SCOPE)
endfunction()

# Creates (if needed) a `thrust.all.[...]` metatarget that ties multiple per-config
# targets together. For instance, calling this with `thrust.cpp.cuda.test.foo` and
# `thrust.cpp.tbb.test.foo` will create a `thrust.all.test.foo` target that can be
# used to build both configurations of the `foo` test.
function(thrust_add_all_config_metatarget target_name)
  string(
    REGEX REPLACE
    "^thrust\\.[^.]+\\.[^.]+\\."
    "thrust.all."
    all_target_name
    "${target_name}"
  )

  # Do nothing if the target doesn't match the expected "thrust.host.device." pattern.
  if (target_name STREQUAL all_target_name)
    return()
  endif()

  if (NOT TARGET ${all_target_name})
    add_custom_target(${all_target_name})
    set_target_properties(${all_target_name} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
  add_dependencies(${all_target_name} ${target_name})
endfunction()
