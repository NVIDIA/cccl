#include <thrust/version.h>

#include <iostream>

int main()
{
  const int major    = THRUST_MAJOR_VERSION;
  const int minor    = THRUST_MINOR_VERSION;
  const int subminor = THRUST_SUBMINOR_VERSION;
  const int patch    = THRUST_PATCH_NUMBER;

  std::cout << "Thrust v" << major << "." << minor << "." << subminor << "-" << patch << '\n';

  return 0;
}
