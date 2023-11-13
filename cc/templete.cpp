#include <iostream>
#include <string>

namespace Ahri {
/**
 * @brief template class
 * @tparam T
 */
template <typename T>
class Person {
private:
    T _height;
    std::string _name;

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

}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::Person<int> person("Ahri", 2);
    person.print_info();
    return 0;
}
