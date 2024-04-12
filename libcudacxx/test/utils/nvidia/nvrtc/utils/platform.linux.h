#pragma once

#include <unistd.h>

static void platform_exec(char const* process, char** args, size_t)
{
  execvp(process, args);
}
