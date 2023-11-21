// lambda (C++11)
// [外部变量访问方式说明符(=, &, this, args,...)](arg) mutable noexcept/throw() -> return type { body };

#include <algorithm>
#include <iostream>

#ifdef __cpp_lambdas
#endif
#ifdef __cpp_generic_lambdas
#endif
int main(int argc, char* argv[]) {
    int c = 100;
    auto func = [&](int a, int b) mutable {
        c = 20;
        return a + b + c;
    };
    std::cout << func(1, 2) << std::endl;
    std::cout << c << std::endl;

    int arr[5]{3, 4, 2, 5, 1};
    std::sort(arr, arr + 5, [=](int x, int y) -> bool { return x < y; });
    for (auto&& n : arr) {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    auto exc = []() noexcept(false) { throw 10; };
    auto exc2 = []() noexcept { return 10; };
    try {
        // exc();
        std::cout << exc2() << std::endl;
    } catch (int) {
        std::cout << "catch int" << std::endl;
    }

    return 0;
}
