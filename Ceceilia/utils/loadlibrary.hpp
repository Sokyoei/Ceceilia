#pragma once
#ifndef LOADLIBRARY_HPP
#define LOADLIBRARY_HPP

#include <string>

#ifdef _WIN32
#include <Windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

#include <fmt/core.h>

namespace Ahri {
class LoadLibrary {
public:
    LoadLibrary(std::string library_name) : _library_name(library_name) {
#ifdef _WIN32
        _platform_library_name = fmt::format("{}.dll", _library_name);
#elif defined(__linux__)
        _platform_library_name = fmt::format("lib{}.so", _library_name);
        _handle = dlopen(_platform_library_name.c_str(), RTLD_LAZY);
#endif
    }

    ~LoadLibrary() {
#ifdef _WIN32
#elif defined(__linux__)
        dlclose(_handle);
#endif
    }

private:
    std::string _library_name;
    std::string _platform_library_name;
#ifdef _WIN32
#elif defined(__linux__)
    void* _handle;
#endif
};
}  // namespace Ahri

#endif  // !LOADLIBRARY_HPP
