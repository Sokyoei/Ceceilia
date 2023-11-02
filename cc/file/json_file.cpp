#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    system("chcp 65001");
#endif
    auto json_file = R"(../data/Ahri/Ahri.json)";
    std::ifstream f(json_file);
    auto settings = nlohmann::json::parse(f);
    std::cout << settings << std::endl;

    return 0;
}
