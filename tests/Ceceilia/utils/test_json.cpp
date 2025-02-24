#include "Ceceilia/utils/json.hpp"

int main(int argc, char const* argv[]) {
    const char* str = "42";
    auto a = Ahri::parse(str);
    std::string str2 = "42";
    auto b = Ahri::parse(str2);
    return 0;
}
