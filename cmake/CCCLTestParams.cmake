# Further documentation and examples are provided in docs/cccl/development/testing.rst.

# The function below reads the filepath `src`, extracts the %PARAM% comments,
# and fills `all_variant_labels_var` with a list of `label1_value1.label2_value2...`
# strings, and puts the corresponding colon-separated definitions into
# `all_variant_defs_var`.
#
# Scalar parameters use colon-separated alternatives:
#   // %PARAM% DEFINITION label value1:value2
#
# Tuple parameters use colon-separated named alternatives and provide multiple
# correlated definitions per alternative. Each name is followed by its
# comma-separated definition values:
#   // %PARAM% DEF1,DEF2 label name1=value1,value2:name2=value3,value4

function(
  cccl_parse_variant_params
  src
  num_variants_var
  all_variant_labels_var
  all_variant_defs_var
)
  file(READ "${src}" file_data)
  # Capture the definition list, target-label stem, and alternatives from each
  # `%PARAM%` line as the first, second, and third match groups, respectively.
  set(param_regex "//[ ]+%PARAM%[ ]+([^ ]+)[ ]+([^ ]+)[ ]+([^\n]*)")

  string(REGEX MATCHALL "${param_regex}" matches "${file_data}")

  set(variant_labels)
  set(variant_defs)

  foreach (match IN LISTS matches)
    string(REGEX MATCH "${param_regex}" unused "${match}")

    set(defs "${CMAKE_MATCH_1}")
    set(label ${CMAKE_MATCH_2})
    set(values "${CMAKE_MATCH_3}")
    string(REPLACE "," ";" defs "${defs}")
    list(LENGTH defs num_defs)

    set(param_labels)
    set(param_defs)
    if (num_defs EQUAL 1)
      string(REPLACE ":" ";" values "${values}")
      foreach (value IN LISTS values)
        list(APPEND param_labels ${label}_${value})
        list(APPEND param_defs ${defs}=${value})
      endforeach()
    else()
      string(STRIP "${values}" values)
      string(REPLACE ":" ";" tuple_values "${values}")

      foreach (tuple_value IN LISTS tuple_values)
        # Split a named tuple alternative at its first '=' into its target-label
        # suffix and comma-separated definition values.
        if (NOT tuple_value MATCHES "^([^=]+)=(.+)$")
          message(
            FATAL_ERROR
            "Malformed tuple parameter '${tuple_value}' in ${src}; expected name=value1,value2"
          )
        endif()

        set(tuple_label "${CMAKE_MATCH_1}")
        set(tuple_elements "${CMAKE_MATCH_2}")
        string(REPLACE "," ";" tuple_elements "${tuple_elements}")
        list(LENGTH tuple_elements num_tuple_elements)
        if (NOT num_tuple_elements EQUAL num_defs)
          message(
            FATAL_ERROR
            "Tuple parameter '${tuple_value}' in ${src} has ${num_tuple_elements} values, but ${num_defs} definitions were provided"
          )
        endif()

        set(tuple_defs)
        math(EXPR tuple_end "${num_defs} - 1")
        foreach (tuple_idx RANGE ${tuple_end})
          list(GET defs ${tuple_idx} tuple_def)
          list(GET tuple_elements ${tuple_idx} tuple_element)
          list(APPEND tuple_defs "${tuple_def}=${tuple_element}")
        endforeach()
        list(JOIN tuple_defs ":" tuple_defs)

        list(APPEND param_labels "${label}_${tuple_label}")
        list(APPEND param_defs "${tuple_defs}")
      endforeach()
    endif()

    # Build lists of test name suffixes (labels) and preprocessor definitions
    # containing the cartesian product of all parameters.
    if (NOT variant_labels)
      set(variant_labels "${param_labels}")
    else()
      set(tmp_labels)
      foreach (old_label IN LISTS variant_labels)
        foreach (param_label IN LISTS param_labels)
          list(APPEND tmp_labels ${old_label}.${param_label})
        endforeach()
      endforeach()
      set(variant_labels "${tmp_labels}")
    endif()

    if (NOT variant_defs)
      set(variant_defs "${param_defs}")
    else()
      set(tmp_defs)
      foreach (old_def IN LISTS variant_defs)
        foreach (param_def IN LISTS param_defs)
          list(APPEND tmp_defs "${old_def}:${param_def}")
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
