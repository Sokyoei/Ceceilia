#include <iostream>
#include <memory>
#include <string>

namespace Ahri {
class Dog {
private:
    struct Impl;
    std::unique_ptr<Impl> _impl;

public:
    Dog();
    ~Dog();
    void print_info() const;
};
}  // namespace Ahri
