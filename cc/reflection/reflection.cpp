/**
 * @file reflection.cpp
 * @date 2025/02/21
 * @author Sokyoei
 * @details
 *  C++ Reflection
 */

#include "config.h"

/// static reflection「静态反射」
/// dynamic reflection「动态反射」
/// compile time reflection「编译期反射」
/// runtime reflection「运行期反射」
namespace Ahri {
/// Run Time Type Information
#if __cpp_rtti
#endif

#ifdef AHRI_CXX26
void cxx26_reflection() {
    constexpr std::meta::info value = ^int;  // 类型映射到值
    using Int = typename[:value:];
    typename[:value:] a = 3;
}
#endif
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    return 0;
}
