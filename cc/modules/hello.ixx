#ifndef __cpp_modules
#error "compiler C++20 module are not support"
#endif  // __cpp_modules

export module hello;

export auto say() {
    return "hello C++20 module";
}
