cmake_minimum_required(VERSION 3.31)  # CMake 3.28+ support C++ Modules

project(Ceceilia VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

########################################################################################################################
# options
########################################################################################################################
option(boost_learn      "build boost_learn subproject"      OFF)
option(folly_learn      "build folly_learn subproject"      OFF)
option(protobuf_learn   "build protobuf_learn subproject"   OFF)
option(drogon_learn     "build drogon_learn subproject"     OFF)

########################################################################################################################
# global settings
########################################################################################################################
add_compile_definitions(AHRI_EXPORT)
# add_definitions(-DAHRI_EXPORT)

include(${PROJECT_SOURCE_DIR}/cmake/compiler.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/library.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)

set(ROOT ${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

########################################################################################################################
# Ahri::Cecceilia and Ahri::Cecceilia-Header-Only
########################################################################################################################
file(
    GLOB_RECURSE
    VISION_SOURCE_FILES
    "src/*.cpp"
    "include/*.hpp"
    "include/*.h"
    "Ceceilia/*.hpp"
    "Ceceilia/*.h"
)
add_library(ahri_ceceilia SHARED ${VISION_SOURCE_FILES})
target_link_libraries(ahri_ceceilia PUBLIC $<$<BOOL:${USE_FMT}>:fmt::fmt>)
target_link_libraries(ahri_ceceilia PUBLIC $<$<BOOL:${USE_SPDLOG}>:spdlog::spdlog>)
target_link_libraries(ahri_ceceilia PUBLIC $<$<BOOL:${USE_NLOHMANN_JSON}>:nlohmann_json::nlohmann_json>)
target_link_libraries(ahri_ceceilia PUBLIC $<$<BOOL:${USE_TINYXML2}>:tinyxml2::tinyxml2>)
# target_link_libraries(ahri_ceceilia PUBLIC $<$<BOOL:${USE_TOMLPLUSPLUS}>:PkgConfig::tomlplusplus>)
target_link_libraries(ahri_ceceilia PUBLIC $<$<BOOL:${USE_TOMLPLUSPLUS}>:tomlplusplus::tomlplusplus>)
target_link_libraries(ahri_ceceilia PUBLIC $<$<BOOL:${USE_YAML_CPP}>:yaml-cpp::yaml-cpp>)
target_compile_definitions(ahri_ceceilia PUBLIC AHRI_CECEILIA_BUILD_SHARED_LIBRARY)
add_library(Ahri::Ceceilia ALIAS ahri_ceceilia)

add_library(ahri_ceceilia_header_only INTERFACE)
target_include_directories(ahri_ceceilia_header_only INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
add_library(Ahri::Ceceilia-Header-Only ALIAS ahri_ceceilia_header_only)

########################################################################################################################
# Subprojects
########################################################################################################################
add_subdirectory(cc)
add_subdirectory(benchmark)
if(CMAKE_CUDA_COMPILER)
    add_subdirectory(cuda)
endif(CMAKE_CUDA_COMPILER)

if(boost_learn AND Boost_FOUND)
    add_subdirectory(boost_learn)
endif(boost_learn AND Boost_FOUND)

if(drogon_learn AND Drogon_FOUND)
    add_subdirectory(drogon_learn)
endif(drogon_learn AND Drogon_FOUND)

if(folly_learn AND folly_FOUND)
    add_subdirectory(folly_learn)
endif(folly_learn AND folly_FOUND)

if(protobuf_learn AND Protobuf_FOUND)
    add_subdirectory(protobuf_learn)
endif(protobuf_learn AND Protobuf_FOUND)

# System
if(WIN32)
    add_subdirectory(windows)
elseif(LINUX)
    add_subdirectory(linux)
endif()

# add_subdirectory(boost)
# add_subdirectory(dll)

# MATRIX
# find_package(Eigen3 CONFIG REQUIRED)
# if (Eigen3_FOUND)
#     include_directories(${EIGEN3_INCLUDE_DIRS})
#     add_subdirectory(Eigen)
# endif ()

#include_directories("D:\\ffmpeg-5.1.2-full_build-shared\\include")
#link_directories("D:\\ffmpeg-5.1.2-full_build-shared\\lib")
#set(FFMPEG_LIBS avcodec avdevice avfilter avformat avutil postproc swresample swscale)
# find_package(FFMPEG REQUIRED)
# if (FFMPEG_FOUND)
#     include_directories(${FFMPEG_INCLUDE_DIRS})
#     add_subdirectory(FFmpeg)
# endif ()

#include_directories("D:\\lua5.4.2\\include")
#link_directories("D:\\lua5.4.2")
# find_package(Lua REQUIRED)
# if (Lua_FOUND)
#     include_directories(${LUA_INCLUDE_DIR})
#     add_subdirectory(Lua)
# endif ()

# add_subdirectory(Magick)

# set(OpenCV_DIR "D:\\opencv4.5.5\\opencv\\build\\x64\\vc15\\lib")
# set(OpenCV_DIR "D:\\Advance\\install\\x64\\vc16\\lib")
# find_package(OpenCV CONFIG REQUIRED)
# if (OpenCV_FOUND)
#     include_directories(${OpenCV_INCLUDE_DIRS})
#     add_subdirectory(OpenCV)
# endif ()

# Python Extension
# find_package(Python COMPONENTS Interpreter Development)
# if (Python_FOUND)
#     add_subdirectory(Python)

#     find_package(pybind11 CONFIG)
#     if (pybind11_FOUND)
#         add_subdirectory(pybind11)
#     endif ()
# endif ()

# add_subdirectory(STL)
# add_subdirectory(Windows)
# add_subdirectory(Sokyoei)
# add_subdirectory(Nono)

if(GTest_FOUND)
    add_subdirectory(tests)
endif(GTest_FOUND)

# config.h settings
configure_file(
    ${PROJECT_SOURCE_DIR}/config.h.cmake
    ${PROJECT_SOURCE_DIR}/config.h
)

# VSCode CMake Tools plugin settings set
# ```json
# "cmake.copyCompileCommands": "${workspaceFolder}/compile_commands.json"
# ```
# copy_compile_commands()

include(${PROJECT_SOURCE_DIR}/cmake/install.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/package.cmake)
