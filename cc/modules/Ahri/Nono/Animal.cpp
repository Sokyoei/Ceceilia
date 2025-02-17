#include "Animal.hpp"

Animal::Animal(int age, std::string name) : _age(age), _name(name) {}

Animal::~Animal() {}

void Animal::eat() {
    std::cout << "Animal can eat." << '\n';
}

std::ostream& operator<<(std::ostream& os, const Animal& animal) {
    os << "name: " << animal._name << ", age: " << animal._age;
    return os;
}
