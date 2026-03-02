# Ceceilia

This is Sokyoei's C/C++/CUDA tutorials and utils project

## Clone Project

```shell
git clone https://github.com/Sokyoei/Ceceilia --recursive
```

## build

### CMake

config

```shell
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake
```

## library

全局安装

```shell
vcpkg install boost folly fmt spdlog nlohmann-json tomlplusplus yaml-cpp tinyxml2 gtest drogon[yaml,orm,sqlite3]
```
