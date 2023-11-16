#include "Pimpl.hpp"

namespace Ahri {
struct Person::Impl {
    std::string name_{"Furina"};
    int age_{16};
    std::string id_{"1030"};

    void print_info() const {
        std::cout << "I'm " << name_ << ", " << age_ << " year old, and My id is " << id_ << std::endl;
    }
};

Person::Person() : impl_(std::make_unique<Impl>()) {}

void Person::print_info() const {
    impl_->print_info();
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::Person p;
    p.print_info();
    return 0;
}
