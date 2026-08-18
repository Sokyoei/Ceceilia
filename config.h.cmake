#ifndef CONFIG_H
#define CONFIG_H

#cmakedefine CECEILIA_ROOT "@CECEILIA_ROOT@"

#define CMAKE

#ifdef __cplusplus
#include "Ahri/Sokyoei.hpp"
#else
#include "Ahri/Sokyoei.h"
#endif  // __cplusplus

// CUDA support
#cmakedefine CECEILIA_HAVE_CUDA

// third libraries
#cmakedefine CECEILIA_USE_FMT
#cmakedefine CECEILIA_USE_SPDLOG
#cmakedefine CECEILIA_USE_NLOHMANN_JSON
#cmakedefine CECEILIA_USE_TOMLPLUSPLUS
#cmakedefine CECEILIA_USE_TINYXML2
#cmakedefine CECEILIA_USE_YAML_CPP
#cmakedefine CECEILIA_USE_GTEST
#cmakedefine CECEILIA_USE_MOSQUITTO
#cmakedefine CECEILIA_USE_PROTOBUF
#cmakedefine CECEILIA_USE_BOOST
#cmakedefine CECEILIA_USE_BOOST_ASIO
#cmakedefine CECEILIA_USE_BENCHMARK
#cmakedefine CECEILIA_USE_FOLLY
#cmakedefine CECEILIA_USE_PROXY
#cmakedefine CECEILIA_USE_MSFT_PROXY4
#cmakedefine CECEILIA_USE_ABSL
#cmakedefine CECEILIA_USE_DROGON
#cmakedefine CECEILIA_USE_JEMALLOC
#cmakedefine CECEILIA_USE_MIMALLOC

#endif  // !CONFIG_H
