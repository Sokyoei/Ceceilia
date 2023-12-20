/**
 * @file timer.cpp
 * @date 2023/12/20
 * @author Sokyoei
 * @details
 * C++ time
 */

#include <chrono>
#include <iostream>

#include "config.h"

#if __cpp_lib_chrono
#endif  // __cpp_lib_chrono

/**
 * @details 时间类型的用户定义字面量
 * C++14(h,min,s,ms,ns,us)
 * C++20(y,d)
 */
#if __cpp_lib_chrono_udls
using namespace std::chrono_literals;
#endif  // __cpp_lib_chrono_udls

namespace Ahri {
/**
 * @brief 时钟
 */
void clocks() {
#ifdef CXX11
#ifdef _MSC_VER
    std::cout << std::chrono::system_clock::now() << std::endl;
// std::cout << std::chrono::steady_clock::now() << std::endl;
// std::cout << std::chrono::high_resolution_clock::now() << std::endl;
#endif  // MSVC

#ifdef CXX20
#ifdef _MSC_VER
    std::cout << std::chrono::gps_clock::now() << std::endl;
    std::cout << std::chrono::utc_clock::now() << std::endl;
    std::cout << std::chrono::tai_clock::now() << std::endl;
    std::cout << std::chrono::file_clock::now() << std::endl;
#endif  // MSVC
#endif  // CXX20
#endif  // CXX11
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    std::cout << std::chrono::system_clock::now() << std::endl;
    Ahri::clocks();
    return 0;
}
