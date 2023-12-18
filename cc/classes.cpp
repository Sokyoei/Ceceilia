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
