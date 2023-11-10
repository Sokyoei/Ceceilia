#include <filesystem>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include "config.h"

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    system("chcp 65001");
#endif
    auto json_file = std::filesystem::path(ROOT) / "data/Ahri/Ahri.json";
    std::ifstream f(json_file);
    auto settings = nlohmann::json::parse(f);
    std::cout << settings << std::endl;

    return 0;
}
