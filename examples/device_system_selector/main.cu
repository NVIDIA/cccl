#include <iostream>
#include <thrust/device_vector.h>

int main()
{
    thrust::device_vector<double> v(100, 7);
    v[3] = 8;
    std::cout << "Element is " << v[3] << "\n";
    return 0;
}
