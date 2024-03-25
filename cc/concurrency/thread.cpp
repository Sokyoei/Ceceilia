/**
 * @file thread.cpp
 * @date 2024/03/22
 * @author Sokyoei
 *
 *
 */

#include <iostream>
#include <string>
#include <string_view>
#include <thread>

namespace Ahri {
void say_hello(std::string_view& str) {
    std::cout << "thread: hello " << str << std::endl;
}

/**
 * 仿函数
 */
class Functor {
public:
    void operator()(std::string_view str) { std::cout << "Functor: " << str << std::endl; }
};

class ClassInner {
public:
    void class_inner(std::string_view str) { std::cout << "class_inner: " << str << std::endl; }
};

void say_good(std::unique_ptr<std::string_view> str) {
    std::cout << "smart_ptr: good " << *str.get() << std::endl;
}

/**
 * 会调用 std::thread 的移动构造函数
 * GCC 编译错误
 * @return std::thread
 */
#ifndef __GNUG__
std::thread func() {
    std::string_view hello = "hello";
    return std::thread(say_good, std::ref(hello));
}
#endif
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    // std::thread 传递参数是拷贝传参
    std::string_view str = "hello";
    // 参数为引用类型，需要使用 std::ref() 显示转换
    std::thread t(Ahri::say_hello, std::ref(str));
    t.join();

    std::thread tfunctor{Ahri::Functor(), str};
    tfunctor.join();

    std::thread tlambda([](std::string_view str) { std::cout << "lambda: " << str << std::endl; }, str);
    tlambda.join();

    // 线程调用类内成员函数，成员函数需要加&，后面第一个参数是类对象，之后才是类成员函数的参数
    Ahri::ClassInner class_inner;
    std::thread ti(&Ahri::ClassInner::class_inner, &class_inner, str);
    ti.join();

    // 函数参数为智能指针类型，需要使用 std::move() 显示传递
    auto good = std::make_unique<std::string_view>("good");
    std::thread tu(Ahri::say_good, std::move(good));
    tu.join();

    // std::thread 通过 std::move() 将所有权转移，转移后 t 无效
    // std::thread tt = std::move(t);

    return 0;
}
