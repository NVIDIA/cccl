enable_language(CUDA)

if (TARGET libcudacxx::libcudacxx)
  # CUDA may not have been enabled when libcudacxx was found:
  libcudacxx_update_language_compat_flags()
endif()

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
