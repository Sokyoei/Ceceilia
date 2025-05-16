/**
 * @file strings.cpp
 * @date 2023/12/20
 * @author Sokyoei
 * @details
 * C++ string
 */

#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

#include "Ceceilia.hpp"

namespace Ahri {
using namespace std::string_literals;
using namespace std::string_view_literals;

const char c[] = "ahri";
#ifdef AHRI_CXX20
const char8_t c8[] = u8"ahri";  // C++20
#endif
const char16_t c16[] = u"ahri";
const char32_t c32[] = U"ahri";
const wchar_t w[] = L"ahri";

std::string s = "ahri"s;
#ifdef AHRI_CXX20
std::u8string u8s = u8"ahri"s;
#endif
std::u16string u16s = u"ahri"s;
std::u32string u32s = U"ahri"s;
std::wstring ws = L"ahri"s;

std::string_view sv = "ahri"sv;
#ifdef AHRI_CXX20
std::u8string_view u8sv = u8"ahri"sv;
#endif
std::u16string_view u16sv = u"ahri"sv;
std::u32string_view u32sv = U"ahri"sv;
std::wstring_view wsv = L"ahri"sv;
}  // namespace Ahri

int main(int argc, char* argv[]) {
/**
 * @brief C++11 原始字符串字面量
 */
#ifdef __cpp_raw_strings
    auto raw_str = R"xxx(Ahri\t\n)xxx";
    std::cout << raw_str << std::endl;
#endif
    return 0;
}
