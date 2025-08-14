/**
 * @file timer.cpp
 * @date 2023/12/20
 * @author Sokyoei
 * @details
 * C++ time
 */

#include "Ahri/Ahri.hpp"

#include <ctime>

#include <chrono>
#include <iomanip>
#include <iostream>

#ifdef _WIN32
#include <Windows.h>
#endif

/**
 * @details 时间类型的用户定义字面量
 * C++14(h,min,s,ms,ns,us)
 * C++20(y,d)
 */
#ifdef __cpp_lib_chrono_udls
using namespace std::chrono_literals;
#endif  // __cpp_lib_chrono_udls

namespace Ahri {
/**
 * @brief 时钟
 */
void clocks() {
#ifdef AHRI_CXX11
#ifdef _MSC_VER
    std::cout << "std::chrono::system_clock::now(): " << std::chrono::system_clock::now() << std::endl;
    // std::cout << "std::chrono::steady_clock::now(): " << std::chrono::steady_clock::now() << std::endl;
    // std::cout << "std::chrono::high_resolution_clock::now(): " << std::chrono::high_resolution_clock::now()
    //           << std::endl;
#endif  // MSVC

#ifdef AHRI_CXX20
#ifdef _MSC_VER
    std::cout << "std::chrono::gps_clock::now(): " << std::chrono::gps_clock::now() << std::endl;
    std::cout << "std::chrono::utc_clock::now(): " << std::chrono::utc_clock::now() << std::endl;
    std::cout << "std::chrono::tai_clock::now(): " << std::chrono::tai_clock::now() << std::endl;
    std::cout << "std::chrono::file_clock::now(): " << std::chrono::file_clock::now() << std::endl;
#endif  // MSVC
#endif  // AHRI_CXX20
#endif  // AHRI_CXX11
}

/**
 * @brief 日期
 */
void date() {
    auto now = std::chrono::system_clock::now();
    std::time_t current_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&current_time);

    std::cout << "当前日期: " << std::put_time(local_time, "%Y/%m/%d") << std::endl;
    std::cout << "当前日期和时间: " << std::put_time(local_time, "%Y/%m/%d %H:%M:%S") << std::endl;

#ifdef AHRI_CXX20
#if __cpp_lib_chrono >= 201907L
    auto localTime = std::chrono::zoned_time{std::chrono::current_zone(), now}.get_local_time();
    auto ymd = std::chrono::year_month_day{std::chrono::floor<std::chrono::days>(localTime)};

    std::cout << "当前日期: " << ymd << std::endl;
#endif
#endif
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    Ahri::clocks();
    Ahri::date();

    return 0;
}
