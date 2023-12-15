/**
 * @file charconvert.hpp
 * @date 2023/12/12
 * @author Sokyoei
 * @details
 * string coding convert
 */

#include <codecvt>
#include <iostream>
#include <locale>
#include <string>

#ifdef _WIN32
#include <Windows.h>
#ifdef _MSC_VER
#include <icu.h>
#endif
#elif defined(__linux__)
#include <iconv.h>
#include <unistd.h>
#endif
