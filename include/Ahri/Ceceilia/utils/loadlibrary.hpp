#pragma once
#ifndef AHRI_CECEILIA_UTILS_LOADLIBRARY_HPP
#define AHRI_CECEILIA_UTILS_LOADLIBRARY_HPP

#include <filesystem>
#include <string>

#ifdef _WIN32
#include <Windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

#include <fmt/core.h>

#include "Ahri/Ceceilia/utils/logger_utils.hpp"

namespace Ahri {
class AhriLoadLibrary  // rename for Windows LoadLibrary macro
{
public:
    AhriLoadLibrary(std::string library_name) : _library_name(library_name) {
#ifdef _WIN32
        _platform_library_name = fmt::format("{}.dll", _library_name);
        handle = LoadLibraryEx(std::filesystem::path(_platform_library_name).wstring().c_str(), nullptr,
                               LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!handle) {
            DWORD error = GetLastError();
            AHRI_LOGGER_ERROR("Failed to load plugin library. Error: {}", error);
            return 1;
        } else {
            AHRI_LOGGER_INFO("Load {} success", _platform_library_name);
        }
#elif defined(__linux__)
        _platform_library_name = fmt::format("lib{}.so", _library_name);
        handle = dlopen(_platform_library_name.c_str(), RTLD_LAZY);
#endif
    }

    ~AhriLoadLibrary() {
#ifdef _WIN32
        FreeLibrary(handle);
#elif defined(__linux__)
        dlclose(handle);
#endif
    }

private:
    std::string _library_name;
    std::string _platform_library_name;

public:
#ifdef _WIN32
    HMODULE handle;
#elif defined(__linux__)
    void* handle;
#endif
};
}  // namespace Ahri

#endif  // !AHRI_CECEILIA_UTILS_LOADLIBRARY_HPP
