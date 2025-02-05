/**
 * @file toml_file.cpp
 * @date 2024/03/22
 * @author Sokyoei
 *
 *
 */

#include "config.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#ifdef USE_TOMLPLUSPLUS
#include <toml++/toml.h>
#elif defined(USE_TOML11)
#include <toml.hpp>
#else
#error "require toml library"
#endif

#include "Ceceilia.hpp"

using namespace std::string_literals;
using namespace std::string_view_literals;

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    std::system("chcp 65001");  // Windows 终端修改代码页以显示 UTF8 字符
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
