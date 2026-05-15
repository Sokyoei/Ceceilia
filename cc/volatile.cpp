/**
 * @file volatile.cpp
 * @date 2026/03/17
 * @author Sokyoei
 * keyword: volatile
 *
 */

#include <iostream>

namespace Ahri {
void volatile_func() {
    volatile int a = 10;
    a += 20;
    a += 30;
    std::cout << a << '\n';
}
}  // namespace Ahri

int main(int argc, char* argv[]) {
    Ahri::volatile_func();

    return 0;
}
