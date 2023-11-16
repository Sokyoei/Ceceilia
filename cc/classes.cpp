#include <iostream>

namespace Ahri {
class Person {
private:
    friend class Sister;

public:
    Person() {}
    ~Person() {}
};
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    return 0;
}
