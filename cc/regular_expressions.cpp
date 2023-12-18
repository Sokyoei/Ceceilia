/**
 * @file regular_expressions.cpp
 * @date 2023/12/18
 * @author Sokyoei
 * @details
 * C++ Regular Expressions
 */

#include <iostream>
#include <regex>
#include <string>

int main(int argc, char const* argv[]) {
    std::string str = "Popstar Ahri";
    std::regex reg{"Ahr.", std::regex::icase};
    std::smatch m;
    if (std::regex_search(str, m, reg)) {
        std::cout << m[0] << std::endl;
    }
    return 0;
}
