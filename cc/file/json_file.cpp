/**
 * @file json_file.cpp
 * @date 2024/03/22
 * @author Sokyoei
 *
 *
 */

#include "config.h"

#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#else
#error "require json library"
#endif

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    system("chcp 65001");
#endif
    auto json_file_path = std::filesystem::path(ROOT) / "data/Ahri/Ahri.json";

#ifdef USE_NLOHMANN_JSON
    std::ifstream f(json_file_path);
    auto settings = nlohmann::json::parse(f);
    std::cout << settings << std::endl;
#endif
    return 0;
}
