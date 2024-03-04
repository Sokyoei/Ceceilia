/**
 * @file classes.cpp
 * @date 2023/12/18
 * @author Sokyoei
 * @details
 * C++ class struct
 */

#include <iostream>
#include <memory>
#include <string>

namespace Ahri {
/**
 * C++ class 会默认有以下成员函数
 */
class A {
public:
    A() {}                                       // Default constructor         默认构造函数
    ~A() {}                                      // Destructor                  析构函数
    A(const A& a) {}                             // Copy constructor            复制构造函数
    A& operator=(const A& a) {}                  // Copy assignment operator    复制赋值运算符
    A(A&& a) noexcept {}                         // Move constructor            移动构造函数
    A& operator=(A&& a) noexcept {}              // Move assignment operator    移动赋值运算符
    A* operator&() { return this; }              // Get address operator        取地址运算符
    const A* operator&() const { return this; }  // Const get address operator  const 取地址运算符
};

class AbstractAnimal {
private:
    int age;
    bool is_alive;

public:
    AbstractAnimal(int age, bool is_alive) : age(age), is_alive(is_alive) {}
    virtual ~AbstractAnimal() {}
    virtual void eat() = 0;
};

class Fox : public AbstractAnimal {
private:
    std::string name;

public:
    Fox(std::string name, int age, int is_alive) : name(name), AbstractAnimal(age, is_alive) {}
    ~Fox() {}
    void eat() override { std::cout << "Fox can eat" << std::endl; }
};

class Person : public AbstractAnimal {
private:
    std::string name;

public:
    Person(std::string name, int age, int is_alive) : name(name), AbstractAnimal(age, is_alive) {}
    ~Person() {}
    void eat() override { std::cout << "Person can eat" << std::endl; }
};

void eat(AbstractAnimal& animal) {
    animal.eat();
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::Fox fox{"Ahri", 18, true};
    Ahri::Person person{"Tom", 21, true};
    eat(fox);
    eat(person);
    return 0;
}
