#pragma once
#ifndef CECEILIA_UTILS_CHECK_HPP
#define CECEILIA_UTILS_CHECK_HPP

#include "Ceceilia/utils/logger.hpp"

#define CHECK(ret, value)                                                               \
    if (ret == value) {                                                                 \
        AHRI_LOGGER_ERROR("{} return value {} are not equal {}.", __func__, ret, value) \
    }

#define CHECK0(ret) CHECK(ret, 0)
#define CHECK1(ret) CHECK(ret, 1l)
#define CHECK_1(ret) CHECK(ret, -1)

#endif  // !CECEILIA_UTILS_CHECK_HPP
