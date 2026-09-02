#
# Architecture options:
#

option(
  CUB_ENABLE_RDC_TESTS
  "Enable tests that require separable compilation."
  ON
)
option(
  CUB_FORCE_RDC
  "Enable separable compilation on all targets that support it."
  OFF
)

option(
  CUB_ENABLE_LAUNCH_NO_LAUNCHER
  "Enable tests/examples without explicit launch variants."
  ON
)
option(
  CUB_ENABLE_LAUNCH_HOST_LAUNCHER
  "Enable host launch variants (lid_0)."
  ON
)
option(
  CUB_ENABLE_LAUNCH_DEVICE_LAUNCHER
  "Enable device launch variants (lid_1)."
  ON
)
option(
  CUB_ENABLE_LAUNCH_GRAPH_LAUNCHER
  "Enable graph launch variants (lid_2)."
  ON
)

option(
  CUB_ENABLE_LAUNCH_VARIANTS
  "Deprecated: use CUB_ENABLE_LAUNCH_DEVICE_LAUNCHER/GRAPH_LAUNCHER to control lid_1/lid_2."
  ON
)

if (NOT CUB_ENABLE_LAUNCH_VARIANTS)
  message(
    WARNING
    "CUB_ENABLE_LAUNCH_VARIANTS is deprecated; disabling lid_1/lid_2 launch variants."
  )
  set(
    CUB_ENABLE_LAUNCH_DEVICE_LAUNCHER
    OFF
    CACHE BOOL
    "Enable device launch variants (lid_1)."
    FORCE
  )
  set(
    CUB_ENABLE_LAUNCH_GRAPH_LAUNCHER
    OFF
    CACHE BOOL
    "Enable graph launch variants (lid_2)."
    FORCE
  )
endif()
