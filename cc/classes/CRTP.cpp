#include <iostream>

namespace Ahri {
// 基类模板
template <typename Derived>
class Base {
public:
    void interface() {
        // 调用子类实现的函数
        static_cast<Derived*>(this)->implementation();
    }

    // 基类版本的函数实现
    void implementation() { std::cout << "Base implementation\n"; }
};

// 子类继承并传递自己作为模板参数
class Derived1 : public Base<Derived1> {
public:
    // 子类版本的函数实现
    void implementation() { std::cout << "Derived1 implementation\n"; }
};

// 另一个子类
class Derived2 : public Base<Derived2> {
public:
    // 子类版本的函数实现
    void implementation() { std::cout << "Derived2 implementation\n"; }
};
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::Derived1 d1;
    Ahri::Derived2 d2;

    // 调用基类中的接口，实际调用的是子类的实现
    d1.interface();  // 输出: Derived1 implementation
    d2.interface();  // 输出: Derived2 implementation

    return 0;
}
