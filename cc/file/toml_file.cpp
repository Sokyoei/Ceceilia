/**
 * @file toml_file.cpp
 * @date 2024/03/22
 * @author Sokyoei
 *
 *
 */

#include "Ahri/Ceceilia.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <Windows.h>
#endif

#ifdef USE_TOMLPLUSPLUS
#if __has_include(<toml++/toml.h>)  // for vcpkg
#include <toml++/toml.h>
#elif __has_include(<toml.hpp>)
#include <toml.hpp>
#endif
#elif defined(USE_TOML11)
#include <toml.hpp>
#else
#error "require toml library"
#endif

using namespace std::string_literals;
using namespace std::string_view_literals;

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    auto toml_file_path = std::filesystem::path(SOKYOEI_DATA_DIR) / "Ahri/Ahri.toml";

#ifdef USE_TOMLPLUSPLUS
    auto settings = toml::parse_file(toml_file_path.c_str());
    std::cout << settings << std::endl;
    std::cout << settings["Ahri Skins"]["Ahri"]["zh-cn"].value_or(""sv) << std::endl;

#elif defined(USE_TOML11)
    auto settings = toml::parse(toml_file_path);
    std::cout << settings << std::endl;

    auto ahri_skins = toml::find(settings, "Ahri Skins");
    auto ahri = toml::find(ahri_skins, "Ahri");
    auto zh_cn = toml::find<std::string>(ahri, "zh-cn");
    std::cout << zh_cn << std::endl;
#endif
    return 0;
}
