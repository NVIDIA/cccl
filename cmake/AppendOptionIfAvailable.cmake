include_guard(GLOBAL)
include(CheckCXXCompilerFlag)

macro(append_option_if_available _FLAG _LIST)
  string(MAKE_C_IDENTIFIER "CXX_FLAG_${_FLAG}" _VAR)
  check_cxx_compiler_flag(${_FLAG} ${_VAR})

  if (${${_VAR}})
    list(APPEND ${_LIST} ${_FLAG})
  endif()
endmacro()
