/**
 * @file type_inference.cpp
 * @date 2023/12/13
 * @author Sokyoei
 * @details
 * C++ 类型推导
 */

#include <iostream>

#include "Ceceilia.hpp"

namespace Ahri {
#if AHRI_CXX14 && __cpp_decltype_auto
template <typename U, typename V>
decltype(auto) add(U& u, V& v) {
    return u + v;
}
#elif defined(AHRI_CXX11)
template <typename U, typename V>
auto add(U& u, V& v) -> decltype(u + v) {
    return u + v;
}
#endif  // AHRI_CXX14

}  // namespace Ahri

int main(int argc, char const* argv[]) {
    int i = 1;
    int* i_ptr = &i;
    int& i_ref = i;
    const int ci = 2;
    volatile int vi = 3;

    // auto
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

    return 0;
}
