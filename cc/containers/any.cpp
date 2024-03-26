/**
 * @file any.cpp
 * @date 2024/03/26
 * @author Sokyoei
 *
 *
 */

#include <any>
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << std::boolalpha;
    std::any a = 1;
    std::cout << a.type().name() << ": " << std::any_cast<int>(a) << '\n';
    std::any b = true;
    std::cout << b.type().name() << ": " << std::any_cast<bool>(b) << '\n';

    try {
        std::cout << std::any_cast<float>(a) << '\n';
    } catch (const std::bad_any_cast& e) {
        std::cout << e.what() << '\n';
    }

    if (a.has_value()) {
        std::cout << a.type().name() << '\n';
    }

    int* i = std::any_cast<int>(&a);
    std::cout << *i << '\n';

    a.reset();
    if (!a.has_value()) {
        std::cout << "no value\n";
    }
    return 0;
}
