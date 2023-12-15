/**
 * @file dump.hpp
 * @date 2023/12/13
 * @author Sokyoei
 * @details
 * Create Windows dump file
 */

#ifndef DUMP_HPP
#define DUMP_HPP

#include <iostream>
#include <string>

#ifdef _WIN32
#include <Windows.h>

#include <DbgHelp.h>
#include <tchar.h>

#pragma comment(lib, "dbghelp.lib")
#elif defined(__linux__)

#else
#error "Dump file are not support"
#endif

namespace Ahri {
class Dump {
private:
    std::string filepath_;

public:
    Dump() {}
    ~Dump() {}
};
}  // namespace Ahri

#endif  // !DUMP_HPP
