# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Add regexes matching deprecated headers here to disable warnings for them:
set(deprecated_headers_regexes "thrust/iterator/tabulate_output_iterator\\.h")

# Meta target for all configs' header builds:
add_custom_target(thrust.all.headers)

find_package(CUDAToolkit)

function(thrust_add_header_test thrust_target label definitions)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_dialect ${thrust_target} DIALECT)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)
  set(config_systems ${config_host} ${config_device})

  string(TOLOWER "${config_host}" host_lower)
  string(TOLOWER "${config_device}" device_lower)

  if (config_device STREQUAL "CUDA")
    set(langs CUDA)
  else()
    # Compile headers with both host and cuda compilers for CPU backends.
    set(langs CXX CUDA)
  endif()

  # GLOB ALL THE THINGS
  set(headers_globs thrust/*.h)
  set(headers_exclude_systems_globs thrust/system/*/*)
  set(
    headers_systems_globs
    thrust/system/${host_lower}/*
    thrust/system/${device_lower}/*
  )
  set(
    headers_exclude_details_globs
    thrust/detail/*
    thrust/*/detail/*
    thrust/*/*/detail/*
  )

  # Get all .h files...
  file(
    GLOB_RECURSE headers
    RELATIVE "${Thrust_SOURCE_DIR}"
    CONFIGURE_DEPENDS
    ${headers_globs}
  )

  # ...then remove all system specific headers...
  file(
    GLOB_RECURSE headers_exclude_systems
    RELATIVE "${Thrust_SOURCE_DIR}"
    CONFIGURE_DEPENDS
    ${headers_exclude_systems_globs}
  )
  list(REMOVE_ITEM headers ${headers_exclude_systems})

  # ...then add all headers specific to the selected host and device systems back again...
  file(
    GLOB_RECURSE headers_systems
    RELATIVE "${Thrust_SOURCE_DIR}"
    CONFIGURE_DEPENDS
    ${headers_systems_globs}
  )
  list(APPEND headers ${headers_systems})

  # ...and remove all the detail headers (also removing the detail headers from the selected systems).
  file(
    GLOB_RECURSE headers_exclude_details
    RELATIVE "${Thrust_SOURCE_DIR}"
    CONFIGURE_DEPENDS
    ${headers_exclude_details_globs}
  )
  list(REMOVE_ITEM headers ${headers_exclude_details})

  # List of headers that aren't implemented for all backends, but are implemented for CUDA.
  set(partially_implemented_CUDA)

  # List of headers that aren't implemented for all backends, but are implemented for CPP.
  set(partially_implemented_CPP)

  # List of headers that aren't implemented for all backends, but are implemented for TBB.
  set(partially_implemented_TBB)

  # List of headers that aren't implemented for all backends, but are implemented for OMP.
  set(partially_implemented_OMP)

  # List of all partially implemented headers.
  set(
    partially_implemented
    ${partially_implemented_CUDA}
    ${partially_implemented_CPP}
    ${partially_implemented_TBB}
    ${partially_implemented_OMP}
  )
  list(REMOVE_DUPLICATES partially_implemented)

  # Filter the partially implemented headers:
  set(headers_tmp ${headers})
  set(headers)
  foreach (header IN LISTS headers_tmp)
    if ("${header}" IN_LIST partially_implemented)
      # This header is partially implemented on _some_ backends...
      if (NOT "${header}" IN_LIST partially_implemented_${config_device})
        # ...but not on the selected one.
        continue()
      endif()
    endif()
    list(APPEND headers ${header})
  endforeach()

  foreach (lang IN LISTS langs)
    set(headertest_target ${config_prefix}.headers.${label})
    if (lang STREQUAL "CUDA" AND (NOT config_device STREQUAL "CUDA"))
      # Append .cuda to the header test target name when compiling
      # CPU backends with cuda compilers.
      set(headertest_target ${headertest_target}.cuda)
    endif()

    cccl_generate_header_tests(
      ${headertest_target}
      thrust
      LANGUAGE ${lang}
      HEADERS ${headers}
      PER_HEADER_DEFINES
        DEFINE
        CCCL_IGNORE_DEPRECATED_API
        ${deprecated_headers_regexes}
    )
    cccl_configure_target(${headertest_target} DIALECT ${config_dialect})
    target_link_libraries(${headertest_target} PUBLIC ${thrust_target})
    thrust_clone_target_properties(${headertest_target} ${thrust_target})
    if (definitions)
      target_compile_definitions(${headertest_target} PRIVATE ${definitions})
    endif()

    if (lang STREQUAL "CUDA")
      thrust_configure_cuda_target(${headertest_target} RDC ${THRUST_FORCE_RDC})
    endif()

    if ("TBB" IN_LIST config_systems)
      # Disable macro checks on TBB; the TBB atomic implementation uses `I` and
      # our checks will issue false errors.
      target_compile_definitions(
        ${headertest_target}
        PRIVATE CCCL_IGNORE_HEADER_MACRO_CHECKS
      )

      # error #550-D: variable "alloc" was set but never used (in TBB headers)
      # Only very specific configs are emitting this:
      if (
        lang STREQUAL "CUDA"
        AND
          "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}"
            VERSION_EQUAL
            "12.0"
        AND MSVC_VERSION EQUAL 1939
        AND config_dialect EQUAL 20
      )
        target_compile_options(
          ${headertest_target}
          PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 550>
        )
      endif()
    endif()

    add_dependencies(thrust.all.headers ${headertest_target})
    add_dependencies(${config_prefix}.all ${headertest_target})
  endforeach()
endfunction()

foreach (thrust_target IN LISTS THRUST_TARGETS)
  thrust_add_header_test(${thrust_target} base "")

  # Wrap Thrust/CUB in a custom namespace to check proper use of ns macros:
  set(
    header_definitions
    "THRUST_WRAPPED_NAMESPACE=wrapped_thrust"
    "CUB_WRAPPED_NAMESPACE=wrapped_cub"
  )
  thrust_add_header_test(${thrust_target} wrap "${header_definitions}")

  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  if ("CUDA" STREQUAL "${config_device}")
    # Check that BF16 support can be disabled
    set(header_definitions "CCCL_DISABLE_BF16_SUPPORT")
    thrust_add_header_test(${thrust_target} no_bf16 "${header_definitions}")

    # Check that half support can be disabled
    set(header_definitions "CCCL_DISABLE_FP16_SUPPORT")
    thrust_add_header_test(${thrust_target} no_half "${header_definitions}")
  endif()
endforeach()
