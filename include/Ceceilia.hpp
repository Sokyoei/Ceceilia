/**
 * @file Ceceilia.hpp
 * @date 2025/02/05
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef CECEILIA_HPP
#define CECEILIA_HPP

#include <cstdlib>
#include <iostream>
#include <string>

#include "../config.h"

namespace Ahri {
std::string _get_sokyoei_data_dir() {
    auto const SOKYOEI_DATA_DIR = std::getenv("SOKYOEI_DATA_DIR");
    if (SOKYOEI_DATA_DIR) {
        return std::string(SOKYOEI_DATA_DIR);
    } else {
        std::cerr << "Please install https://github.com/Sokyoei/data" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static auto const SOKYOEI_DATA_DIR = _get_sokyoei_data_dir();
}  // namespace Ahri

#define SOKYOEI_DATA_DIR Ahri::SOKYOEI_DATA_DIR

#endif  // !CECEILIA_HPP
