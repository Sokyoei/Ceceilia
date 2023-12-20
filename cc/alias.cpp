/**
 * @file alias.cpp
 * @date 2023/12/20
 * @author Sokyoei
 * @details
 * C/C++ alias
 */

#include <array>
#include <iostream>

namespace Ahri {
// C typedef
typedef int INT;
typedef int (*int_any_fn_t)();
typedef int (*int_Nx10x10_t)[10][10];

// C++ using
using FLOAT = float;
using float_fn_t = float (*)();
using float_10x10_t = std::array<std::array<float, 10>, 10>;

#ifdef __cpp_alias_templates

#endif  // __cpp_alias_templates
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    return 0;
}
