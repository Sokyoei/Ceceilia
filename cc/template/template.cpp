/**
 * @file template.cpp
 * @date 2023/12/20
 * @author Sokyoei
 * @details
 * C++ template
 */

#include <iostream>
#include <string>

namespace Ahri {
/**
 * @brief template function
 * @tparam T
 * @param t
 */
template <class T>
void say(T t) {
    std::cout << t << '\n';
}

/**
 * @brief template class
 * @tparam T
 */
template <typename T>
class Person {
private:
    std::string _name;
    T _height;

public:
    Person(std::string name, T height);
    ~Person();
    void print_info();
};

template <typename T>
Person<T>::Person(std::string name, T height) : _name(std::move(name)), _height(height) {}

template <typename T>
Person<T>::~Person() {}

template <typename T>
void Person<T>::print_info() {
    std::cout << "I'm " << _name << " height: " << _height << std::endl;
}

/**
 * @brief template lambda
 * @tparam T
 */
template <typename T>
auto f = [](const T& t) { std::cout << t << '\n'; };

/**
 * @brief template global const value
 * @tparam T
 */
template <typename T>
constexpr inline int major_version = 1;

/**
 * 模板特化（完全特化/部分特化）是为模板提供一个特定类型的（完全/部分）实现
 * @tparam
 */
template <>
class Person<int> {
private:
    std::string _name;
    int _height;

public:
    Person(std::string name, int height);
    ~Person();
    void print_info();
};

Person<int>::Person(std::string name, int height) : _name(std::move(name)), _height(height) {}

Person<int>::~Person() {}

void Person<int>::print_info() {
    std::cout << "I'm " << _name << " height: " << _height << std::endl;
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::say<int>(2);
    Ahri::Person<int> person("Ahri", 2);
    person.print_info();
    Ahri::f<std::string>("hello world");
    return 0;
}
