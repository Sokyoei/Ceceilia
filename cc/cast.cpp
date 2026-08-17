#include <iostream>
#include <string>

namespace Ahri {
class Animal {
public:
    Animal() {}

    virtual ~Animal() {}

    void fun() { std::cout << "Animal" << std::endl; }
};

class Fox : public Animal {
    void fun() { std::cout << "Fox" << std::endl; }
};

// const_cast: only const_cast remove (const volatile or __unaligned)
void ahri_const_cast() {
    const std::string s = "ahri";
    std::string& r = const_cast<std::string&>(s);
    std::cout << r << std::endl;
    std::string* p = const_cast<std::string*>(&s);
    std::cout << *p << std::endl;
}

// dynamic_cast: runtime check, only cast ptr or reference, return nullptr if cast failed
void ahri_dynamic_cast() {
    Animal* animal = new Fox;
    Fox* fox_ptr = dynamic_cast<Fox*>(animal);
    if (fox_ptr == nullptr) {
        std::cout << "dynamic cast fail" << std::endl;
    } else {
        std::cout << "dynamic cast success" << std::endl;
    }
    delete animal;
}

// reinterpret_cast: execute low-cast
void ahri_reinterpret_cast() {
    Animal animal;
    Fox* fox_ptr = reinterpret_cast<Fox*>(&animal);
    if (fox_ptr == nullptr) {
        std::cout << "reinterpret cast fail" << std::endl;
    } else {
        std::cout << "reinterpret cast success" << std::endl;
    }
}

// static_cast: like C cast, no-runtime check
void ahri_static_cast() {
    std::cout << static_cast<int>(3.14) << std::endl;
    std::string s = "ahri";
    std::string& r = static_cast<std::string&>(s);
    std::cout << r << std::endl;
    std::string* p = static_cast<std::string*>(&s);
    std::cout << *p << std::endl;
}
}  // namespace Ahri

int main(int argc, char* argv[]) {
    Ahri::ahri_const_cast();
    Ahri::ahri_dynamic_cast();
    Ahri::ahri_reinterpret_cast();
    Ahri::ahri_static_cast();

    return 0;
}
