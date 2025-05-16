/**
 * @file function_pointer.cpp
 * @date 2023/12/22
 * @author Sokyoei
 * @details
 * C++ function pointer
 */

// std::function
// std::invoke

#include <functional>
#include <iostream>
#include <string>

#include "Ceceilia.hpp"

namespace Ahri {
void hello() {
    std::cout << "hello, Furina" << std::endl;
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    // use function pointer
#ifdef AHRI_CXX11
    using void_void_fn_t = void (*)();
#elif defined(AHRI_CXX98)
    typedef void (*void_void_fn_t)();
#endif  // AHRI_CXX11
    void_void_fn_t func = Ahri::hello;
    func();

    // use std::function
    std::function<void()> fn = Ahri::hello;
    fn();

#ifdef __cpp_lib_invoke
    // use std::invoke
    std::invoke(&Ahri::hello);
#endif

    return 0;
}
