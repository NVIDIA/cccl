file(READ "${TEMPLATE_PRE}" _pre)
file(READ "${JIT_TEMPLATE}" _jit)
file(READ "${TEMPLATE_POST}" _post)

file(WRITE "${OUT}" "${_pre}")
file(APPEND "${OUT}" "${_jit}")
file(APPEND "${OUT}" "${_post}")
