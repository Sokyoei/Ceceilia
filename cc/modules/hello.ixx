/**
 * @file hello.ixx
 * @date 2023/12/13
 * @author Sokyoei
 * @details
 * C++20 module
 */

#ifndef __cpp_modules
#error "compiler C++20 module are not support"
#endif  // __cpp_modules

export module hello;

export auto say() -> const char* {
    return "hello C++20 module";
}
