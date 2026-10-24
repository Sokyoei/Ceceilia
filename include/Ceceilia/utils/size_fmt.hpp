#pragma once
#ifndef CECEILIA_UTILS_SIZE_FMT_HPP
#define CECEILIA_UTILS_SIZE_FMT_HPP

#include <array>
#include <cstddef>
#include <string>

#include <fmt/core.h>
#include <fmt/format.h>

namespace Ceceilia {
inline std::string size_fmt(size_t bytes) {
    const constexpr std::array<const char*, 5> units{"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    auto size = static_cast<double>(bytes);
    while (size >= 1024 && unit_idx < 4) {
        size /= 1024;
        unit_idx++;
    }
    return fmt::format("{:.2f}{}", size, units[unit_idx]);
}
}  // namespace Ceceilia

#endif  // !CECEILIA_UTILS_SIZE_FMT_HPP
