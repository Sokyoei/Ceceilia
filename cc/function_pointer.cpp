// std::function
// std::invoke

#include <functional>
#include <iostream>
#include <string>

#include "config.h"

namespace Ahri {
void hello() {
    std::cout << "hello, Furina" << std::endl;
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    // use function pointer
#ifdef CXX11
    using void_void_fn_t = void (*)();
#elif defined(CXX98)
    typedef void (*void_void_fn_t)();
#endif  // CXX11
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
