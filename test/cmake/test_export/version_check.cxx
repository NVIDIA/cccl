// Compile with:
//   -DVERSION_HEADER=include/path/for/version.h
//   -DEXPECTED_VERSION=XXYYZZ
//   -DVERSION_MACRO=PROJECT_VERSION

#define HEADER <VERSION_HEADER>
#include HEADER

#include <cstdio>

#define DETECTED_VERSION VERSION_MACRO

int main()
{
  printf("Expected version: %d\n"
         "Detected version: %d\n",
         EXPECTED_VERSION,
         VERSION_MACRO);
  return EXPECTED_VERSION == DETECTED_VERSION ? 0 : 1;
}
