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
#include <utility>

#include "Ahri/Ahri.hpp"

namespace Ahri {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief C++ 三五法则
/// 如果类自定义了下面任意一个成员函数，就必须手动定义另外两（四）个
/// - @b 析构函数           三法则、五法则
/// - @b 复制构造函数       三法则、五法则
/// - @b 复制赋值运算符     三法则、五法则
/// - @b 移动构造函数       五法则
/// - @b 移动赋值运算符     五法则
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 *  @brief C++ class 会默认有以下成员函数
 */
class A {
public:
    /// ----------------------------------------------------------------------------------------------------------------
    /// @brief Default constructor 默认构造函数
    A() = default;
    /// @brief Destructor 析构函数
    ~A() = default;
    /// @brief Copy constructor 复制构造函数
    A(const A& a) = default;
    /// @brief Copy assignment operator 复制赋值运算符
    A& operator=(const A& a) {
        if (this != &a) {
            _value = a._value;
        }
        return *this;
    }
    /// @brief Move constructor 移动构造函数
    A(A&& a) noexcept = default;
    /// @brief Move assignment operator 移动赋值运算符
    A& operator=(A&& a) noexcept {
        if (this != &a) {
            _value = a._value;
            a._value = 0;
        }
        return *this;
    }
    /// @brief Get address operator 取地址运算符
    A* operator&() { return this; }
    /// @brief Const get address operator const 取地址运算符
    const A* operator&() const { return this; }

    /// ----------------------------------------------------------------------------------------------------------------
    /// @brief Constructor with parameter 带参数构造函数
    explicit A(int value) : _value(value) {};
    /// @brief Get/Set functions 获取/设置值函数
    AHRI_NODISCARD int get_value() const { return _value; }
    void set_value(int value) { _value = value; }

private:
    int _value;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class B {
public:
    /// @brief 内联成员函数（隐式）
    void implicit_inline_function() {}
    /// @brief 内联成员函数（显式）声明
    inline void explicit_inline_function();
};

/// @brief 内联成员函数（显式）定义，必须跟声明在同一文件
inline void B::explicit_inline_function() {}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief 继承与多态
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief 抽象动物基类
 * @details
 * C++ 不能声明为虚函数的函数包括：
 *  1、普通函数（非成员函数）
 *  2、构造函数
 *  3、内联成员函数
 *  4、静态成员函数
 *  5、友元函数
 */
class AbstractAnimal {
private:
    int _age;
    bool _is_alive;

public:
    AbstractAnimal(int age, bool is_alive) : _age(age), _is_alive(is_alive) { std::cout << "AbstractAnimal()" << '\n'; }
    /// @brief 基类指针指向派生类时，基类的析构函数需要为虚函数，否则派生类析构时它析构函数不能被正常调用
    /* virtual */ ~AbstractAnimal() { std::cout << "~AbstractAnimal()" << '\n'; }
    /// @brief 纯虚函数，派生类必须实现
    virtual void eat() = 0;
    // virtual inline void drink() = 0;
};

class Fox : public AbstractAnimal {
private:
    std::string _name;
    int* _a;

public:
    Fox(std::string name, int age, bool is_alive)
        : _name(std::move(name)), _a(new int(3)), AbstractAnimal(age, is_alive) {
        std::cout << "Fox()" << '\n';
    }
    ~Fox() {
        std::cout << "~Fox()" << '\n';
        delete _a;
    }
    void eat() override { std::cout << "Fox can eat" << '\n'; }
};

class Person : public AbstractAnimal {
private:
    std::string _name;

public:
    Person(std::string name, int age, bool is_alive) : _name(std::move(name)), AbstractAnimal(age, is_alive) {
        std::cout << "Person()" << '\n';
    }
    ~Person() { std::cout << "~Person()" << '\n'; }
    void eat() override { std::cout << "Person can eat" << '\n'; }
};

void eat(AbstractAnimal& animal) {
    animal.eat();
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    // Ahri::Fox fox{"Ahri", 18, true};
    // Ahri::Person person{"Tom", 21, true};
    // Ahri::AbstractAnimal* animal;
    // // 基类指针指向派生类以实现多态
    // animal = &fox;
    // animal->eat();
    // animal = &person;
    // animal->eat();

    // eat(fox);
    // eat(person);

    std::unique_ptr<Ahri::AbstractAnimal> a = std::make_unique<Ahri::Fox>("Ahri", 18, true);
    // Ahri::AbstractAnimal* a = new Ahri::Fox{"Ahri", 18, true};
    // delete a;

    return 0;
}
