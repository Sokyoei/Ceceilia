/**
 * @file yaml_file.cpp
 * @date 2024/03/22
 * @author Sokyoei
 *
 *
 */

#include "config.h"

#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef USE_YAML_CPP
#include <yaml-cpp/yaml.h>
#else
#error "require yaml library"
#endif

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    system("chcp 65001");
#endif
    auto yaml_file_path = std::filesystem::path(ROOT) / "data/Ahri/Ahri.yaml";

#ifdef USE_YAML_CPP
    std::fstream f(yaml_file_path);
    auto yaml_file = YAML::Load(f);
    std::cout << yaml_file << std::endl;
#endif
    return 0;
}
