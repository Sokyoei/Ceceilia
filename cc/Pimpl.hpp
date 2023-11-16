#include <iostream>
#include <memory>
#include <string>

namespace Ahri {
class Person {
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

public:
    Person();
    ~Person() = default;
    void print_info() const;
};
}  // namespace Ahri
