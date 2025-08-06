/**
 * @file string.hpp
 * @date 2025/01/20
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef AHRI_CECELIA_CORE_STRING_HPP
#define AHRI_CECELIA_CORE_STRING_HPP

#include <string>

#ifdef _WIN32
#include <Windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace Ahri {
#define DEFAULE_STRING_SIZE 20

enum class StringEncodingType {
    GBK,
    UTF8,
    UTF16,
};

class String {
public:
    String(char* str) {
        _data = new char[DEFAULE_STRING_SIZE];
        _size = DEFAULE_STRING_SIZE;
        _capacity = DEFAULE_STRING_SIZE;
    }

    String(wchar_t* str) {}
    template <typename STLType>
    String(STLType& str) {}
    ~String() {}

    bool operator<=>(const String& other) const {}

    template <typename STLType>
    STLType to_stl_string() {}

private:
    char* _data;
    size_t _size;
    size_t _capacity;
};

class StringView {
public:
    StringView() {}
    ~StringView() {}
};

template <typename SrcType, typename DstType>
DstType convert(StringEncodingType from, StringEncodingType to, SrcType src) {
#ifdef _WIN32

#elif defined(__linux__)

#endif
}
}  // namespace Ahri

#endif  // !AHRI_CECELIA_CORE_STRING_HPP
