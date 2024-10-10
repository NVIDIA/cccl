set(CCCL_EXECUTABLE_OUTPUT_DIR "${CCCL_BINARY_DIR}/bin")
set(CCCL_LIBRARY_OUTPUT_DIR "${CCCL_BINARY_DIR}/lib")

# Setup common properties for all test/example/etc targets.
function(cccl_configure_target target_name)
  set(options)
  set(oneValueArgs DIALECT)
  set(multiValueArgs)
  cmake_parse_arguments(CCT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  get_target_property(type ${target_name} TYPE)

  if (DEFINED CCT_DIALECT)
    set_target_properties(${target_name}
      PROPERTIES
        CXX_STANDARD ${CCT_DIALECT}
        CUDA_STANDARD ${CCT_DIALECT}
        # Must manually request that the standards above are actually respected
        # or else CMake will silently fail to configure the targets correctly...
        # Note that this doesn't actually work as of CMake 3.16:
        # https://gitlab.kitware.com/cmake/cmake/-/issues/20953
        # We'll leave these properties enabled in hopes that they will someday
        # work.
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD_REQUIRED ON
    )

    get_property(langs GLOBAL PROPERTY ENABLED_LANGUAGES)
    set(dialect_features)
    if (CUDA IN_LIST langs)
      list(APPEND dialect_features cuda_std_${CCT_DIALECT})
    endif()
    if (CXX IN_LIST langs)
      list(APPEND dialect_features cxx_std_${CCT_DIALECT})
    endif()

    get_target_property(type ${target_name} TYPE)
    if (${type} STREQUAL "INTERFACE_LIBRARY")
      target_compile_features(${target_name} INTERFACE
        ${dialect_features}
      )
    else()
      target_compile_features(${target_name} PUBLIC
        ${dialect_features}
      )
    endif()
  endif()

  if (NOT ${type} STREQUAL "INTERFACE_LIBRARY")
    set_target_properties(${target_name}
      PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CCCL_LIBRARY_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CCCL_LIBRARY_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CCCL_EXECUTABLE_OUTPUT_DIR}"
    )
  endif()
endfunction()
