enable_language(CUDA)

#
# Architecture options:
#

option(
  THRUST_ENABLE_RDC_TESTS
  "Enable tests that require separable compilation."
  ON
)
option(
  THRUST_FORCE_RDC
  "Enable separable compilation on all targets that support it."
  OFF
)
