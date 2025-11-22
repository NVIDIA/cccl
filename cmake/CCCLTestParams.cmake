# Further documentation and examples are provided in docs/cccl/development/testing.rst.

# cccl_detect_test_variants(test_basename test_src
#   [PREFIX <prefix>] # default: "variant"
# )
#
# Detects %PARAM% variants in the given source file `test_src` and prepares
# the corresponding variant names and definitions for use.
#
# The detected variant info is stored in variables with the given PREFIX, which
# defaults to "variant" if not provided:
#
# - <PREFIX>_NUM_VARIANTS: number of test variants
# - <PREFIX>_KEYS: list of variant keys
# - For each variant key:
#   - <KEY>_NAME: the full test name for this variant
#   - <KEY>_LABEL: the variant label (suffix)
#   - <KEY>_DEFINITIONS: the list of preprocessor definitions for this variant
#
# If no variants are detected, a single variant with no definitions is created, and
# <PREFIX>_NO_VARIANT_PARAMS is set to TRUE. Otherwise, it is set to FALSE.
#
# A VAR_IDX definition is always added to the variant definitions, containing
# the zero-based index of the variant in the key list.
#
# Example usage:
#
#  cccl_detect_test_variants(my_test_base "path/to/source_file.cpp")
#  foreach (key IN LISTS variant_KEYS)
#    add_executable(${${key}_NAME} "path/to/source_file.cpp")
#    target_compile_definitions(${${key}_NAME} PRIVATE ${${key}_DEFINITIONS})
#    add_test(NAME ${${key}_NAME} COMMAND ${${key}_NAME})
#  endforeach()
#
function(cccl_detect_test_variants test_basename test_src)
  set(options)
  set(oneValueArgs PREFIX)
  set(multiValueArgs)
  cmake_parse_arguments(
    self
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  cccl_parse_arguments_error_checks(
    "cccl_add_executable"
    ERROR_UNPARSED
    DEFAULT_VALUES PREFIX "variant"
  )

  # Detect %PARAM% variants:
  cccl_parse_variant_params(
    "${test_src}"
    variant_count
    variant_labels
    variant_defs
  )

  if (variant_count EQUAL 0)
    set(${self_PREFIX}_NO_VARIANT_PARAMS TRUE PARENT_SCOPE)
    # Create a single variant with no params:
    set(variant_count 1)
    list(APPEND variant_labels "")
    list(APPEND variant_defs "")
  else()
    set(${self_PREFIX}_NO_VARIANT_PARAMS FALSE PARENT_SCOPE)
    cccl_log_variant_params(
      ${test_basename}
      ${variant_count}
      variant_labels
      variant_defs
    )
  endif()

  # Subtract 1 to support the inclusive endpoint of foreach(...RANGE...):
  math(EXPR variant_range_end "${variant_count} - 1")
  set(variant_keys)
  foreach (var_idx RANGE ${variant_range_end})
    cccl_get_variant_data(variant_labels variant_defs ${var_idx} label defs)

    if (label STREQUAL "")
      set(suffix)
    else()
      set(suffix ".${label}")
    endif()

    set(key "${self_PREFIX}_key_${var_idx}")
    set(${key}_NAME "${test_basename}${suffix}" PARENT_SCOPE)
    set(${key}_LABEL "${label}" PARENT_SCOPE)
    set(${key}_DEFINITIONS "${defs}" PARENT_SCOPE)
    list(APPEND variant_keys "${key}")
  endforeach() # Variant
  set(${self_PREFIX}_NUM_VARIANTS "${variant_count}" PARENT_SCOPE)
  set(${self_PREFIX}_KEYS "${variant_keys}" PARENT_SCOPE)
endfunction()

# The function below reads the filepath `src`, extracts the %PARAM% comments,
# and fills `all_variant_labels_var` with a list of `label1_value1.label2_value2...`
# strings, and puts the corresponding `DEFINITION=value1:DEFINITION=value2`
# entries into `all_variant_defs_var`.
function(
  cccl_parse_variant_params
  src
  num_variants_var
  all_variant_labels_var
  all_variant_defs_var
)
  set(param_regex "//[ ]+%PARAM%[ ]+([^ ]+)[ ]+([^ ]+)[ ]+([^\n]*)")

  # Cache the %PARAM% matches to avoid re-reading the source file multiple times.
  # This is especially important when multiple tests are added from the same source file,
  # like with multiple Thrust host/device configs.
  get_filename_component(src_absolute "${src}" ABSOLUTE)
  string(MD5 source_filename_md5 "${src_absolute}")
  set(param_cache_property "_cccl_variant_param_cache_${source_filename_md5}")
  get_property(param_cache_set GLOBAL PROPERTY "${param_cache_property}" SET)
  if (param_cache_set)
    get_property(param_cache GLOBAL PROPERTY "${param_cache_property}")
  else()
    file(READ "${src_absolute}" source_contents)
    string(REGEX MATCHALL ${param_regex} param_cache "${source_contents}")
    set_property(GLOBAL PROPERTY "${param_cache_property}" "${param_cache}")
  endif()

  # Changes to the source file should re-run CMake to pick-up new error specs:
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${src_absolute}"
  )

  set(matches "${param_cache}")

  set(variant_labels)
  set(variant_defs)

  foreach (match IN LISTS matches)
    string(REGEX MATCH "${param_regex}" unused "${match}")

    set(def ${CMAKE_MATCH_1})
    set(label ${CMAKE_MATCH_2})
    set(values "${CMAKE_MATCH_3}")
    string(REPLACE ":" ";" values "${values}")

    # Build lists of test name suffixes (labels) and preprocessor definitions
    # (defs) containing the cartesian product of all param values:
    if (NOT variant_labels)
      foreach (value IN LISTS values)
        list(APPEND variant_labels ${label}_${value})
      endforeach()
    else()
      set(tmp_labels)
      foreach (old_label IN LISTS variant_labels)
        foreach (value IN LISTS values)
          list(APPEND tmp_labels ${old_label}.${label}_${value})
        endforeach()
      endforeach()
      set(variant_labels "${tmp_labels}")
    endif()

    if (NOT variant_defs)
      foreach (value IN LISTS values)
        list(APPEND variant_defs ${def}=${value})
      endforeach()
    else()
      set(tmp_defs)
      foreach (old_def IN LISTS variant_defs)
        foreach (value IN LISTS values)
          list(APPEND tmp_defs ${old_def}:${def}=${value})
        endforeach()
      endforeach()
      set(variant_defs "${tmp_defs}")
    endif()
  endforeach()

  list(LENGTH variant_labels num_variants)

  set(${num_variants_var} "${num_variants}" PARENT_SCOPE)
  set(${all_variant_labels_var} "${variant_labels}" PARENT_SCOPE)
  set(${all_variant_defs_var} "${variant_defs}" PARENT_SCOPE)
endfunction()

# Extracts the variant label and definitions for the given variant index and prepares them for use.
function(
  cccl_get_variant_data
  all_variant_labels_var
  all_variant_defs_var
  var_idx
  label_var
  defs_var
)
  if ("${${all_variant_labels_var}}" STREQUAL "")
    set(label "")
  else()
    list(GET ${all_variant_labels_var} ${var_idx} label)
  endif()

  if ("${${all_variant_defs_var}}" STREQUAL "")
    set(defs "")
  else()
    list(GET ${all_variant_defs_var} ${var_idx} defs)
  endif()

  string(REPLACE ":" ";" defs "${defs}")
  list(APPEND defs "VAR_IDX=${var_idx}")
  set(${label_var} "${label}" PARENT_SCOPE)
  set(${defs_var} "${defs}" PARENT_SCOPE)
endfunction()

# Logs the detected variant info to CMake's VERBOSE output stream.
function(
  cccl_log_variant_params
  name_base
  num_variants
  all_variant_labels_var
  all_variant_defs_var
)
  # Verbose output:
  if (num_variants GREATER 0)
    message(VERBOSE "Detected ${num_variants} variants of '${name_base}':")

    # Subtract 1 to support the inclusive endpoint of foreach(...RANGE...):
    math(EXPR range_end "${num_variants} - 1")
    foreach (var_idx RANGE ${range_end})
      cccl_get_variant_data(
        ${all_variant_labels_var}
        ${all_variant_defs_var}
        ${var_idx}
        label
        defs
      )
      message(VERBOSE "  ${var_idx}: ${label} ${defs}")
    endforeach()
  endif()
endfunction()
