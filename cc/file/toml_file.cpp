#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#define USE_TOMLPLUSPLUS
// #define USE_TOML11

#ifdef USE_TOMLPLUSPLUS
#include <toml++/toml.h>
#elif defined(USE_TOML11)
#include <toml.hpp>
#else
#error "require toml library"
#endif

#include "config.h"

using namespace std::string_literals;
using namespace std::string_view_literals;

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    std::system("chcp 65001");  // Windows 终端修改代码页以显示 UTF8 字符
#endif

    auto toml_file_path = std::filesystem::path(ROOT) / "data/Ahri/Ahri.toml";

#ifdef USE_TOML11
    auto settings = toml::parse(toml_file_path);  // toml11
    std::cout << settings << std::endl;

    auto ahri_skins = toml::find(settings, "Ahri Skins");
    auto ahri = toml::find(ahri_skins, "Ahri");
    auto zh_cn = toml::find<std::string>(ahri, "zh-cn");
    std::cout << zh_cn << std::endl;
#elif defined(USE_TOMLPLUSPLUS)
    auto settings = toml::parse_file(toml_file_path.c_str());  // tomlplusplus
    std::cout << settings << std::endl;

    std::cout << settings["Ahri Skins"]["Ahri"]["zh-cn"].value_or(""sv) << std::endl;
#endif
    return 0;
}
