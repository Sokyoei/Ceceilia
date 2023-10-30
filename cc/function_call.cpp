// __stdcall  (Standard Call)
// __cdecl    (C Declaration)
// __fastcall
// __thiscall
// __pascal

#include <iostream>
#include <typeinfo>

namespace Ahri {
// C++ 标准调用方式，从右往左依次压入栈，被调函数自己清理堆栈（自动清栈）
// C: _functionname@number
int __stdcall Ahri_stdcall(int a, int b) {
    return a + b;
}

// C 缺省调用方式，从右往左依次压入栈，调用者清理堆栈（手动清栈）
// C: _functionname
extern "C" int __cdecl Ahri_cdecl(int a, int b) {
    return a + b;
}

// 通过 CPU 寄存器传递参数
// C: @functionname@number
int __fastcall Ahri_fastcall(int a, int b) {
    return a + b;
}

// C++ 类成员函数缺省调用方式，从右往左依次压入栈
// 参数个数确定，this 指针通过 ecx 传递给被调用者，被调函数自己清理堆栈
// 参数个数不确定，this 指针在所有参数压入栈后被压入栈，调用者清理堆栈
class Ahri_ThisCall {
public:
    int __thiscall Ahri_thiscall(int a, int b) { return a + b; }
};
}  // namespace Ahri

int main(int argc, char* argv[]) {
    std::cout << typeid(Ahri::Ahri_stdcall).name() << std::endl;
    std::cout << typeid(Ahri::Ahri_cdecl).name() << std::endl;
    std::cout << typeid(Ahri::Ahri_fastcall).name() << std::endl;
    std::cout << typeid(Ahri::Ahri_ThisCall().Ahri_thiscall).name() << std::endl;

    return 0;
}
