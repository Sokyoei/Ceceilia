#include <iostream>

void f() {
    static int a = 1;  // static 改变生命周期，和全局生命周期一样长（一般情况作用域和生命周期一样长）
    std::cout << a << '\n';
    a++;
}

int main() {
    f();
    f();
    f();

    return 0;
}
