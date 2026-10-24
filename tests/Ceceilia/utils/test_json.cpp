#include "Ceceilia/utils/json.hpp"

int main(int argc, char const* argv[]) {
    const char* str = "42";
    auto a = Ceceilia::parse(str);
    std::string str2 = "42";
    auto b = Ceceilia::parse(str2);
    return 0;
}
