// attribute
//
// [[noreturn]]           (C++11) 指示函数不返回
// [[carries_dependency]] (C++11) 指示释放消费 std::memory_order 中的依赖链传入和传出该函数
// [[deprecated]]         (C++14) 指示允许使用声明有此属性的名称或实体，但因 原因 而不鼓励使用
// [[fallthrough]]        (C++17) 指示从前一 case 标号直落是有意的，而在发生直落时给出警告的编译器不应该为此诊断
// [[nodiscard]]          (C++17) 鼓励编译器在返回值被舍弃时发布警告
// [[maybe_unused]]       (C++17) 压制编译器在未使用实体上的警告，若存在
// [[likely]]             (C++20) 指示编译器应该针对通过某语句的执行路径比任何其他执行路径更可能的情况进行优化
// [[unlikely]]           (C++20) 指示编译器应该针对通过某语句的执行路径比任何其他执行路径更不可能的情况进行优化
// [[no_unique_address]]  (C++20) 指示非静态数据成员不需要拥有不同于其类的所有其他非静态数据成员的地址
// [[assume]]             (C++23) Specifies that an expression will always evaluate to true at a given point

// https://learn.microsoft.com/zh-cn/cpp/cpp/attributes?view=msvc-170

#include <iostream>

namespace Ahri {
#if __cpp_attributes
[[gnu::const]] [[nodiscard]] int ahri_nodiscard() {
    return 1;
};

[[nodiscard("别忘记返回值")]] int ahri_nodiscard2() {
    return 1;
};

[[noreturn]] int ahri_noreturn() {
    std::cout << "[[noreturn]]" << std::endl;
    return 1;
}

[[carries_dependency]] int ahri_carries_dependency() {
    return 1;
}

// msvc error c4996
[[deprecated]] int ahri_deprecated() {
    return 1;
}

// msvc error c4996
[[deprecated("已弃用")]] int ahri_deprecated2() {
    return 1;
}

//[[fallthrough]] int ahri_fallthrough() {}

[[maybe_unused]] int ahri_maybe_unused(int a, int b) {
    return 1;
}

//[[likely]] int ahri_likely() {}
//[[unlikely]] int ahri_unlikely() {}
//[[no_unique_address]] int ahri_no_unique_address() {}
//[[assume]] int ahri_assume() {}
//[[optimize_for_synchronized]] int ahri_optimize_for_synchronized() {}
#endif  // __cpp_attributes
}  // namespace Ahri

int main(int argc, char* argv[]) {
    Ahri::ahri_nodiscard();
    Ahri::ahri_nodiscard2();
    std::cout << Ahri::ahri_noreturn() << std::endl;

    // std::cout << Ahri::ahri_deprecated() << std::endl;
    // std::cout << Ahri::ahri_deprecated2() << std::endl;
    std::cout << Ahri::ahri_maybe_unused(1, 1) << std::endl;

    return 0;
}
