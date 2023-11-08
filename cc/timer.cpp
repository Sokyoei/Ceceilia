#include <chrono>
#include <iostream>

#if __cpp_lib_chrono
#endif  // __cpp_lib_chrono

/**
 * C++14(h,min,s,ms,ns,us)
 * C++20(y,d)
 * 时间类型的用户定义字面量
 */
#if __cpp_lib_chrono_udls
using namespace std::chrono_literals;
#endif  // __cpp_lib_chrono_udls

namespace Ahri {
void clocks() {
#if __cplusplus >= 201103L  // C++11
#ifdef _MSC_VER
    std::cout << std::chrono::system_clock::now() << std::endl;
// std::cout << std::chrono::steady_clock::now() << std::endl;
// std::cout << std::chrono::high_resolution_clock::now() << std::endl;
#endif  // MSVC

#if __cplusplus >= 202002L  // C++20
#ifdef _MSC_VER
    std::cout << std::chrono::gps_clock::now() << std::endl;
    std::cout << std::chrono::utc_clock::now() << std::endl;
    std::cout << std::chrono::tai_clock::now() << std::endl;
    std::cout << std::chrono::file_clock::now() << std::endl;
#endif  // MSVC
#endif  // C++20
#endif  // C++11
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    std::cout << std::chrono::system_clock::now() << std::endl;
    Ahri::clocks();
    return 0;
}
