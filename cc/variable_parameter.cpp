/**
 * @file variable_parameter.cpp
 * @date 2023/12/13
 * @author Sokyoei
 * @details
 * variable parameter「可变长参数」
 */

#include "Ahri/Ahri.hpp"

#include <cstdarg>

#include <initializer_list>
#include <iostream>

namespace Ahri {
/**
 * C va_list
 */
extern "C" void
#ifdef _MSC_VER
    __cdecl
#else
    __attribute__((cdecl))
#endif
    c_vparam(int x, ...) {
    va_list ap;
    va_start(ap, x);
    for (int i = 0; i < x; i++) {
        std::cout << va_arg(ap, int) << " ";
    }
    std::cout << std::endl;
    va_end(ap);
}

/**
 * C++11
 * @tparam T
 * @param t
 */
template <typename T>
void cxx11_vparam(T t) {
    std::cout << t << std::endl;
}
template <typename T, typename... Args>
void cxx11_vparam(T t, Args... args) {
    std::cout << t << " ";
    cxx11_vparam(args...);
}

/**
 * std::initializer_list
 * @tparam T
 * @param args
 */
template <typename T>
void cxx_initializer_list(std::initializer_list<T> args) {
    for (auto& i : args) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

// array
template <typename T, typename... Args>
void func3(Args... args) {
    T arr[]{args...};
    for (auto& i : arr) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

#ifdef AHRI_CXX17
#ifdef __cpp_if_constexpr
/**
 * C++17 if constexpr
 * @tparam T
 * @tparam Args
 * @param t
 * @param args
 */
template <typename T, typename... Args>
void cxx17_if_constexpr(T t, Args... args) {
    std::cout << t << " ";
    if constexpr (sizeof...(args) > 0) {
        cxx17_if_constexpr(args...);
    } else {
        std::cout << std::endl;
    }
}
#endif  // __cpp_if_constexpr

#ifdef __cpp_fold_expressions
/**
 * @brief C++17 fold expression
 *
 * @tparam Args
 * @param args
 * @return auto
 */
template <typename... Args>
auto cxx17_fold_expression(Args... args) {
    return (args + ...);
}
#endif  // __cpp_fold_expressions
#endif  // AHRI_CXX17
}  // namespace Ahri

int main(int argc, char* argv[]) {
    Ahri::c_vparam(5, 1, 2, 3, 4, 5);
    Ahri::cxx11_vparam<int>(1, 2, 3, 4, 5);
    Ahri::cxx_initializer_list<int>({1, 2, 3, 4, 5});
    Ahri::func3<int>(1, 2, 3, 4, 5);
    Ahri::cxx17_if_constexpr<int>(1, 2, 3, 4, 5);
    std::cout << Ahri::cxx17_fold_expression(1, 2, 3, 4, 5) << std::endl;
    return 0;
}
