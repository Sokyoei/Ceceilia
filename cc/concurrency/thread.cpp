#include <iostream>
#include <string>
#include <string_view>
#include <thread>

namespace Ahri {
void say_hello(std::string_view& str) {
    std::cout << "ref: hello " << str << std::endl;
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    std::string_view a = "asd";
    std::thread t(Ahri::say_hello, std::ref(a));
    t.join();
    return 0;
}
