# This file defines thrust_configure_multiconfig(), which sets up and handles
# the MultiConfig options that allow multiple host/device configurations
# to be generated from a single thrust build.

function(thrust_configure_multiconfig)
  option(
    THRUST_ENABLE_MULTICONFIG
    "Enable multiconfig options for coverage testing."
    OFF
  )

  if (THRUST_ENABLE_MULTICONFIG)
    # Systems:
    option(
      THRUST_MULTICONFIG_ENABLE_SYSTEM_CPP
      "Generate build configurations that use CPP."
      ON
    )
    option(
      THRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA
      "Generate build configurations that use CUDA."
      ON
    )
    option(
      THRUST_MULTICONFIG_ENABLE_SYSTEM_OMP
      "Generate build configurations that use OpenMP."
      OFF
    )
    option(
      THRUST_MULTICONFIG_ENABLE_SYSTEM_TBB
      "Generate build configurations that use TBB."
      OFF
    )

    # Workload:
    # - `SMALL`: [3 configs] Minimal coverage and validation of each device system against the `CPP` host.
    # - `MEDIUM`: [6 configs] Cheap extended coverage.
    # - `LARGE`: [8 configs] Expensive extended coverage. Include all useful build configurations.
    # - `FULL`: [12 configs] The complete cross product of all possible build configurations.
    #
    # Config   | Workloads | Value      | Expense   | Note
    # ---------|-----------|------------|-----------|-----------------------------
    # CPP/CUDA | F L M S   | Essential  | Expensive | Validates CUDA against CPP
    # CPP/OMP  | F L M S   | Essential  | Cheap     | Validates OMP against CPP
    # CPP/TBB  | F L M S   | Essential  | Cheap     | Validates TBB against CPP
    # CPP/CPP  | F L M     | Important  | Cheap     | Tests CPP as device
    # OMP/OMP  | F L M     | Important  | Cheap     | Tests OMP as host
    # TBB/TBB  | F L M     | Important  | Cheap     | Tests TBB as host
    # TBB/CUDA | F L       | Important  | Expensive | Validates TBB/CUDA interop
    # OMP/CUDA | F L       | Important  | Expensive | Validates OMP/CUDA interop
    # TBB/OMP  | F         | Not useful | Cheap     | Mixes CPU-parallel systems
    # OMP/TBB  | F         | Not useful | Cheap     | Mixes CPU-parallel systems
    # TBB/CPP  | F         | Not Useful | Cheap     | Parallel host, serial device
    # OMP/CPP  | F         | Not Useful | Cheap     | Parallel host, serial device

    set(
      THRUST_MULTICONFIG_WORKLOAD
      SMALL
      CACHE STRING
      "Limit host/device configs: SMALL (up to 3 h/d combos per dialect), MEDIUM(6), LARGE(8), FULL(12)"
    )
    set_property(
      CACHE THRUST_MULTICONFIG_WORKLOAD
      PROPERTY STRINGS SMALL MEDIUM LARGE FULL
    )
    set(
      THRUST_MULTICONFIG_WORKLOAD_SMALL_CONFIGS
      CPP_OMP
      CPP_TBB
      CPP_CUDA
      CACHE INTERNAL
      "Host/device combos enabled for SMALL workloads."
      FORCE
    )
    set(
      THRUST_MULTICONFIG_WORKLOAD_MEDIUM_CONFIGS
      ${THRUST_MULTICONFIG_WORKLOAD_SMALL_CONFIGS}
      CPP_CPP
      TBB_TBB
      OMP_OMP
      CACHE INTERNAL
      "Host/device combos enabled for MEDIUM workloads."
      FORCE
    )
    set(
      THRUST_MULTICONFIG_WORKLOAD_LARGE_CONFIGS
      ${THRUST_MULTICONFIG_WORKLOAD_MEDIUM_CONFIGS}
      OMP_CUDA
      TBB_CUDA
      CACHE INTERNAL
      "Host/device combos enabled for LARGE workloads."
      FORCE
    )
    set(
      THRUST_MULTICONFIG_WORKLOAD_FULL_CONFIGS
      ${THRUST_MULTICONFIG_WORKLOAD_LARGE_CONFIGS}
      OMP_CPP
      TBB_CPP
      OMP_TBB
      TBB_OMP
      CACHE INTERNAL
      "Host/device combos enabled for FULL workloads."
      FORCE
    )

    # Hide the single config options if they exist from a previous run:
    if (DEFINED THRUST_HOST_SYSTEM)
      set_property(CACHE THRUST_HOST_SYSTEM PROPERTY TYPE INTERNAL)
      set_property(CACHE THRUST_DEVICE_SYSTEM PROPERTY TYPE INTERNAL)
    endif()
  else() # Single config:
    # Restore system option visibility if these cache options already exist
    # from a previous run.
    if (DEFINED THRUST_HOST_SYSTEM)
      set_property(CACHE THRUST_HOST_SYSTEM PROPERTY TYPE STRING)
    else()
      set(
        THRUST_HOST_SYSTEM
        "CPP"
        CACHE STRING
        "The targeted host system: ${THRUST_HOST_SYSTEM_OPTIONS}"
      )
      set_property(
        CACHE THRUST_HOST_SYSTEM
        PROPERTY STRINGS ${THRUST_HOST_SYSTEM_OPTIONS}
      )
    endif()
    if (DEFINED THRUST_DEVICE_SYSTEM)
      set_property(CACHE THRUST_DEVICE_SYSTEM PROPERTY TYPE STRING)
    else()
      set(
        THRUST_DEVICE_SYSTEM
        "CUDA"
        CACHE STRING
        "The targeted device system: ${THRUST_DEVICE_SYSTEM_OPTIONS}"
      )
      set_property(
        CACHE THRUST_DEVICE_SYSTEM
        PROPERTY STRINGS ${THRUST_DEVICE_SYSTEM_OPTIONS}
      )
    endif()
  endif()
endfunction()
