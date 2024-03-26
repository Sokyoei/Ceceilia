/**
 * @file concept.cpp
 * @date 2024/03/26
 * @author Sokyoei
 *
 *
 */

#include <concepts>
#include <iostream>
#include <type_traits>

#ifdef __cpp_lib_concepts
#endif

namespace Ahri {
#ifdef __cpp_concepts
template <typename T>
concept c = std::is_same_v<T, int>;
void f(c auto T) {
    std::printf("hello concept\n");
}
#endif  // __cpp_concepts
}  // namespace Ahri

int main(int argc, char* argv[]) {
    Ahri::f(1);
    return 0;
}
