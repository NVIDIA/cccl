set(CCCL_EXECUTABLE_OUTPUT_DIR "${CCCL_BINARY_DIR}/bin")
set(CCCL_LIBRARY_OUTPUT_DIR "${CCCL_BINARY_DIR}/lib")

# Setup common properties for all test/example/etc targets.
function(cccl_configure_target target_name)
  set(options)
  set(oneValueArgs DIALECT)
  set(multiValueArgs)
  cmake_parse_arguments(
    CCT
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  get_target_property(type ${target_name} TYPE)

  set_target_properties(
    ${target_name}
    PROPERTIES
      # Disable compiler extensions:
      CXX_EXTENSIONS OFF
      CUDA_EXTENSIONS OFF
  )

  if (DEFINED CCT_DIALECT)
    set(CMAKE_CXX_STANDARD ${CCT_DIALECT})
    set(CMAKE_CUDA_STANDARD ${CCT_DIALECT})
  endif()

  set_target_properties(
    ${target_name}
    PROPERTIES
      CXX_STANDARD ${CMAKE_CXX_STANDARD}
      CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
      CXX_STANDARD_REQUIRED ON
      CUDA_STANDARD_REQUIRED ON
  )

  get_property(langs GLOBAL PROPERTY ENABLED_LANGUAGES)
  set(dialect_features)
  if (CUDA IN_LIST langs)
    list(APPEND dialect_features cuda_std_${CMAKE_CUDA_STANDARD})
  endif()
  if (CXX IN_LIST langs)
    list(APPEND dialect_features cxx_std_${CMAKE_CXX_STANDARD})
  endif()

  get_target_property(type ${target_name} TYPE)
  if (${type} STREQUAL "INTERFACE_LIBRARY")
    target_compile_features(${target_name} INTERFACE ${dialect_features})
  else()
    target_compile_features(${target_name} PUBLIC ${dialect_features})
  endif()

  if (NOT ${type} STREQUAL "INTERFACE_LIBRARY")
    set_target_properties(
      ${target_name}
      PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CCCL_LIBRARY_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CCCL_LIBRARY_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CCCL_EXECUTABLE_OUTPUT_DIR}"
    )
  endif()
endfunction()
