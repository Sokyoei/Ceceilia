#include "Pimpl.hpp"

namespace Ahri {
struct Dog::Impl {
    std::string _name{"Furina"};
    int _age{16};
    std::string _id{"1030"};

    void print_info() const {
        std::cout << "I'm " << _name << ", " << _age << " year old, and My id is " << _id << std::endl;
    }
};

Dog::Dog() : _impl(std::make_unique<Impl>()) {}

Dog::~Dog() {}

void Dog::print_info() const {
    _impl->print_info();
}
}  // namespace Ahri
