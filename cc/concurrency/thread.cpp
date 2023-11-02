#include <iostream>
#include <thread>

namespace Ahri {
void say_hello() {
    std::cout << "hello world" << std::endl;
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    std::thread t(Ahri::say_hello);
    t.join();
    return 0;
}
