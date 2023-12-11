#include <filesystem>
#include <fstream>
#include <iostream>

#define USE_YAML_CPP

#ifdef USE_YAML_CPP
#include <yaml-cpp/yaml.h>
#else
#error "require yaml library"
#endif

#include "config.h"

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
