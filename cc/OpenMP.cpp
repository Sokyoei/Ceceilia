#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char const* argv[]) {
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < 10; i++) {
        std::cout << i << std::endl;
    }
    return 0;
}
