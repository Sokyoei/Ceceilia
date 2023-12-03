// constexpr(常量表达式)
//
// constexpr 变量
// constexpr 函数
// constexpr 构造函数
// constexpr 析构函数

#include <iostream>

#if __cpp_constexpr
#endif

#if __cpp_if_constexpr
#endif

namespace Ahri {
template <typename T>
constexpr T add(T a, T b) {
    return a + b;
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    constexpr auto hello = "hello world";
    std::cout << hello << std::endl;

    std::cout << Ahri::add<int>(1, 2) << std::endl;

    return 0;
}
