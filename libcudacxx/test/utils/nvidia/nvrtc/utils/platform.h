#pragma once

#if defined(_MSC_VER)
#  define WINDOWS_STUFF
#  include "platform.win.h"
#else
#  include "platform.linux.h"
#endif
