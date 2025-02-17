#pragma once
#ifndef ANIMAL_HPP
#define ANIMAL_HPP

#include <iostream>
#include <string>

class Animal {
public:
    Animal(int age, std::string name);
    ~Animal();
    void eat();
    friend std::ostream& operator<<(std::ostream& os, const Animal& animal);

private:
    int _age;
    std::string _name;
};

#endif  // !ANIMAL_HPP
