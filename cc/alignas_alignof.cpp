/**
 * @file alignas_alignof.cpp
 * @date 2025/02/07
 * @author Sokyoei
 *
 *
 */

#include <iostream>

#if __cpp_alias_templates
#endif
#if __cpp_aligned_new
#endif

struct alignas(16) ahri_alignas_info {
    char a;
    int b;
    double c;
};

int main(int argc, char* argv[]) {
    std::cout << sizeof(ahri_alignas_info) << std::endl;
    return 0;
}
