#include <iostream>

namespace Ahri {
class Fox {
public:
    /*explicit*/ Fox(int age = 0) : age(age) {}

    int getAge() const { return age; }

private:
    int age;
};

void show_info(const Fox& fox) {
    std::cout << "const: Fox(age: " << fox.getAge() << ")" << std::endl;
}
}  // namespace Ahri

int main(int argc, char* argv[]) {
    Ahri::show_info(1);

    return 0;
}
