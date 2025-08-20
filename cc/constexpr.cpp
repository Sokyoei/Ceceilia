/**
 * @file constexpr.cpp
 * @date 2023/12/13
 * @author Sokyoei
 * @details
 * C++11 constexpr「常量表达式」
 */

#include "Ahri/Ahri.hpp"

#include <iostream>
#include <print>

namespace Ahri {
#if __cpp_constexpr >= 200704L && defined AHRI_CXX11
/// @brief 声明编译期常量(constexpr 变量)
constexpr int x = 10;
constexpr auto hello = "hello world";

/// @brief 函数只能有一个 return, 参数和返回值必须是字面量、基本类型、指针、引用等
template <typename T>
constexpr T add(T a, T b) {
    return a + b;
}

#if __cpp_constexpr >= 201304L && defined AHRI_CXX14
/// @brief 允许 constexpr 函数包含局部变量、循环、条件分支
constexpr int factorial(int n) {
    if (n < 0) {
        return 0;
    }
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

/// @brief 支持非 const 的 constexpr 成员函数(成员函数可以修改 *this)
struct Counter {
    int value;
    constexpr void increment() { value++; }
};

#if __cpp_constexpr >= 201603L && defined AHRI_CXX17
/// @brief constexpr lambda
template <int N>
struct Foo {};

void constexpr_lambda() {
    constexpr auto sum = [](int x, int y) constexpr { return x + y; };
    constexpr int z = sum(1, 2);
    std::println("z: {}", z);

    AHRI_MAYBE_UNUSED Foo<sum(2, 3)> foo;  // equal to Foo<5>
}

#if __cpp_constexpr >= 201907L && defined AHRI_CXX20
/// @brief constexpr call virtual function(需要满足编译期多态)
struct Base {
    virtual constexpr int f() const = 0;
    virtual constexpr ~Base() = default;
};

struct Derived : public Base {
    constexpr int f() const override { return 42; }
    constexpr ~Derived() override = default;
};

constexpr Ahri::Derived d;
constexpr const Ahri::Base& c_b_ref = d;

/// @brief constexpr dynamic_cast
constexpr int constexpr_dynamic_cast(const Base& b) {
    if (const auto* d = dynamic_cast<const Derived*>(&b)) {
        return d->f();
    }
    return -1;
}

/// @brief constexpr try/catch
constexpr int safe_divide(int a, int b) {
    try {
        if (b == 0) {
            throw std::runtime_error("Divide by zero");
        }
        return a / b;
    } catch (...) {
        return -1;
    }
}
#if __cpp_constexpr >= 202002L && defined AHRI_CXX20
/// @brief constexpr 修改 union 成员
union U {
    int a;
    float b;
};

constexpr float convert_union(int a) {
    U u;
    u.a = a;
    return u.b;
}

#endif  // __cpp_constexpr >= 202002L && defined AHRI_CXX20
#endif  // __cpp_constexpr >= 201907L && defined AHRI_CXX20
#endif  // __cpp_constexpr >= 201304L && defined AHRI_CXX14
#endif  // __cpp_constexpr >= 201603L && defined AHRI_CXX17
#endif  // __cpp_constexpr >= 200704L && defined AHRI_CXX11

#if defined(__cpp_if_constexpr) && defined(AHRI_CXX17)
#endif  // defined (__cpp_if_constexpr)&&defined (AHRI_CXX17)

#if defined __cpp_constexpr_dynamic_alloc and defined AHRI_CXX20
constexpr int accumulate(int n) {
    if (n <= 0) {
        return 0;
    }

    int* arr = new int[n];

    for (int i = 0; i < n; i++) {
        arr[i] = i + 1;
    }

    int total = 0;
    for (int i = 0; i < n; i++) {
        total += arr[i];
    }

    delete[] arr;

    return total;
}

template <int N>
struct Checker {};

#endif  // defined __cpp_constexpr_dynamic_alloc and defined AHRI_CXX20

#if defined(__cpp_constexpr_in_decltype) && defined(AHRI_CXX20)
#endif  // defined(__cpp_constexpr_in_decltype) && defined(AHRI_CXX20)

#if defined(__cpp_constexpr_exceptions) && defined(AHRI_CXX26)
#endif  // defined (__cpp_constexpr_exceptions) && defined(AHRI_CXX26)
}  // namespace Ahri

int main(int argc, char const* argv[]) {
#if __cpp_constexpr >= 200704L && defined AHRI_CXX11
    std::println("Ahri::hello: {}", Ahri::hello);
    std::println("Ahri::add<int>(1, 2): {}", Ahri::add<int>(1, 2));
    std::println("Ahri::x: {}", Ahri::x);

#if __cpp_constexpr >= 201304L && defined AHRI_CXX14
    std::println("Ahri::factorial(5): {}", Ahri::factorial(5));
    constexpr Ahri::Counter c = [] {
        Ahri::Counter c{0};
        c.increment();
        return c;
    }();
    std::println("c.value: {}", c.value);

#if __cpp_constexpr >= 201603L && defined AHRI_CXX17
    Ahri::constexpr_lambda();

#if __cpp_constexpr >= 201907L && defined AHRI_CXX20
    constexpr int val = Ahri::c_b_ref.f();
    std::println("val: {}", val);

    std::println("Ahri::print_Derived(d): {}", Ahri::constexpr_dynamic_cast(Ahri::c_b_ref));

    std::println("Ahri::safe_divide(10, 2): {}", Ahri::safe_divide(10, 2));
    std::println("Ahri::safe_divide(10, 0): {}", Ahri::safe_divide(10, 0));

#if __cpp_constexpr >= 202002L && defined AHRI_CXX20
    std::println("Ahri::convert_union(10): {}", Ahri::convert_union(10));

#endif  // __cpp_constexpr >= 202002L && defined AHRI_CXX20
#endif  // __cpp_constexpr >= 201907L && defined AHRI_CXX20
#endif  // __cpp_constexpr >= 201603L && defined AHRI_CXX17
#endif  // __cpp_constexpr >= 201304L && defined AHRI_CXX14
#endif  // __cpp_constexpr >= 200704L && defined AHRI_CXX11

#if defined __cpp_constexpr_dynamic_alloc and defined AHRI_CXX20
    constexpr int result = Ahri::accumulate(10);
    std::println("result: {}", result);
    AHRI_MAYBE_UNUSED Ahri::Checker<result> check;

#endif  // defined __cpp_constexpr_dynamic_alloc and defined AHRI_CXX20
    return 0;
}
