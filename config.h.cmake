#ifndef CONFIG_H
#define CONFIG_H

#cmakedefine ROOT "@ROOT@"

#include "include/Sokyoei.h"

// third libraries
#cmakedefine USE_FMT
#cmakedefine USE_SPDLOG
#cmakedefine USE_NLOHMANN_JSON
#cmakedefine USE_TOMLPLUSPLUS
#cmakedefine USE_TINYXML2
#cmakedefine USE_YAML_CPP
#cmakedefine USE_GTEST
#cmakedefine USE_MOSQUITTO
#cmakedefine USE_PROTOBUF
#cmakedefine USE_BOOST_ASIO

#endif  // !CONFIG_H
