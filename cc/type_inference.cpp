/**
 * @file type_inference.cpp
 * @date 2023/12/13
 * @author Sokyoei
 * @details
 * C++ 类型推导
 */

#include "Ahri/Ahri.hpp"

#include <iostream>

namespace Ahri {
#if AHRI_CXX20
auto add(auto x, auto y) {
    return x + y;
}
// 等价于
// template <typename U, typename V>
// auto add(U u, V v) {
//     return u + v;
// }
#elif AHRI_CXX14 && __cpp_decltype_auto
template <typename U, typename V>
decltype(auto) add(U u, V v) {
    return u + v;
}
#elif defined(AHRI_CXX11)
template <typename U, typename V>
auto add(U u, V v) -> decltype(u + v) {
    return u + v;
}
#endif  // AHRI_CXX14 && __cpp_decltype_auto

/// 泛型 lambda
/// @see [Generic lambda expressions](https://wg21.link/N3649)
#if defined(AHRI_CXX14) && defined(__cpp_generic_lambdas) && __cpp_generic_lambdas >= 201304L
auto lambda = [](auto x) { return x + 1; };
#endif

/// 显式模板 lambda
/// @see [Explicit template parameter list for generic lambdas](https://wg21.link/P0428R2)
#if defined(AHRI_CXX20) && defined(__cpp_generic_lambdas) && __cpp_generic_lambdas >= 201707L
auto lambda2 = []<typename T>(T x) { return x + 1; };
#endif

// struct S {
//     auto x = 10;
// };

/// C++17 非类型模板参数 auto
/// @see [Declaring non-type template parameter with auto](https://wg21.link/P0127R2)
#if defined(AHRI_CXX17) && defined(__cpp_nontype_template_parameter_auto) && \
    __cpp_nontype_template_parameter_auto >= 201606L
template <auto T>
void foo() {}
#endif
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    int i = 1;
    int* i_ptr = &i;
    int& i_ref = i;
    const int ci = 2;
    volatile int vi = 3;

    /// auto
    /// 不能在普通函数的参数中直接使用（C++20 放开，简写模板参数）
    /// 不能作用于类的非静态成员变量（C++23 放开）
    /// 不能直接定义数组 auto arr[N]
    /// 不能作用于模板参数（C++17 放开 (template<auto N>)）
    auto a_i = i;
    auto a_i_ptr = i_ptr;
    auto* a_star_i_ptr = i_ptr;
    auto a_i_ref = i_ref;  // 丢弃引用
    auto a_ci = ci;
    auto a_vi = vi;     // 丢弃 cv 限定符
    auto& a_ref_i = i;  // 对 auto 增加限定符以推导为想到的类型

#ifdef __cpp_decltype
    // decltype
    decltype(i) d_i;
    decltype(i_ptr) d_i_ptr;
    decltype(i_ref) d_i_ref = i;
    decltype(ci) d_ci = i;
    decltype(vi) d_vi;
#endif  // __cpp_decltype

    std::cout << Ahri::add<int, float>(1, 2.3) << '\n';
    return 0;
}
