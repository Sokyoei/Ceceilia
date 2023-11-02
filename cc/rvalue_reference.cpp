// rvalue reference (C++11) 右值引用

//            expression
//            /       \
//           /         \
//       glvalue       rvalue
//       广义左值        广义右值
//       /     \       /     \
//      /       \     /       \
//   lvalue      xvalue      prvalue
//    左值        将亡值       纯右值

#include <iostream>

namespace Ahri {
// // C++98 const lvalue reference 常量左值引用
// #if __cplusplus >= 199711L
// const int& rref_cxx98 = 1;

// // int& t 只接收左值，无法接收右值
// void otherdef_cxx98(int& t) {
//     std::cout << "C++98 lvalue\n";
// }

// // const int& t 既能接收左值，也能接收右值；除非被调用的函数参数也是 const 属性，否则无法直接传递
// void otherdef_cxx98(const int& t) {
//     std::cout << "C++98 rvalue\n";
// }

// // C++98 perfect forward 完美转发
// template <typename T>
// void func_cxx98(T& t) {
//     otherdef_cxx98(t);
// }

// template <typename T>
// void func_cxx98(const T& t) {
//     otherdef_cxx98(t);
// }
// #endif  // __cplusplus >= 199711L

/**
 * C++11 右值引用
 */
#ifdef __cpp_rvalue_references
// C++11
#if __cplusplus >= 201103L
int&& rref_cxx11 = 1;

void otherdef_cxx11(int& t) {
    std::cout << "C++11 lvalue\n";
}

void otherdef_cxx11(int&& t) {
    std::cout << "C++11 rvalue\n";
}

// C++11 perfect forward 完美转发
// 一般情况下，&& 只能接收右值；但在函数模板中，&& 既能接收左值，也能接收右值
template <typename T>
void func_cxx11(T&& t) {
    otherdef_cxx11(std::forward<T>(t));
}
#endif  // C++11
#endif  // __cpp_rvalue_references
}  // namespace Ahri

int main(int argc, char* argv[]) {
    // 字符串字面量是左值
    std::cout << &"Ahri" << std::endl;

    // Ahri::func_cxx98(5);
    // int a = 1;
    // Ahri::func_cxx98(a);

    Ahri::func_cxx11(5);
    int b = 1;
    Ahri::func_cxx11(b);

    return 0;
}
