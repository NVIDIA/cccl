# Further documentation and examples are provided in docs/cccl/development/testing.rst.

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
  file(READ "${src}" file_data)
  set(param_regex "//[ ]+%PARAM%[ ]+([^ ]+)[ ]+([^ ]+)[ ]+([^\n]*)")

  string(REGEX MATCHALL "${param_regex}" matches "${file_data}")

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
  list(GET ${all_variant_labels_var} ${var_idx} label)
  list(GET ${all_variant_defs_var} ${var_idx} defs)
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
