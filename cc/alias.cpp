/**
 * @file alias.cpp
 * @date 2023/12/20
 * @author Sokyoei
 * @details
 * C/C++ alias
 */

#include <array>
#include <iostream>
#include <map>

namespace Ahri {
// C typedef
typedef int INT;
typedef int (*int_any_fn_t)();
typedef int (*int_Nx10x10_t)[10][10];
// typedef 不能定义一个模板，需要写在结构体里
template <typename T>
// typedef std::map<std::string, T> str_T;  // 此处不能指定“typedef”
struct str_T {
    typedef std::map<std::string, T> msT;
};
str_T<int>::msT map_str_int;

// C++ using
using FLOAT = float;
using float_fn_t = float (*)();
using float_10x10_t = std::array<std::array<float, 10>, 10>;
// 模板
template <typename T>
using str_T2 = std::map<std::string, T>;

#ifdef __cpp_alias_templates

#endif  // __cpp_alias_templates
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    return 0;
}
