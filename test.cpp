// hardcoded version
#define __NV_TARGET_VERSION_03_01 301

#ifndef __NV_TARGET_VERSION_MAX
#  define __NV_TARGET_FIRST_INCLUDE
#  define __NV_TARGET_VERSION_MAX __NV_TARGET_VERSION_03_01
#elif __NV_TARGET_VERSION_MAX <= __NV_TARGET_VERSION_03_01
#  undef __NV_TARGET_VERSION_MAX
#  define __NV_TARGET_VERSION_MAX __NV_TARGET_VERSION_03_01
#endif

#if defined(__NV_TARGET_FIRST_INCLUDE)
#  undef __NV_TARGET_FIRST_INCLUDE
#  if __has_include(<nv/target>)
#    include <nv/target>
#  endif
#elif __has_include_next(<nv/target>)
#  include_next <nv/target>
#elif __NV_TARGET_VERSION_MAX != __NV_TARGET_VERSION_03_01
#  define __NV_SKIP_THIS_INCLUDE
#endif

#ifndef __NV_SKIP_THIS_INCLUDE

#  ifndef __NV_TARGET_H
#    define __NV_TARGET_H

static_assert(false);

#  endif // __NV_TARGET_H

#else
#  undef __NV_SKIP_THIS_INCLUDE
#endif
