cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/vcpkg/scripts/buildsystems/vcpkg.cmake")

project(Ceceilia VERSION 0.0.1 LANGUAGES C CXX)

#enable_language(CUDA)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

add_compile_definitions(AHRI_EXPORT)
# add_definitions(-DAHRI_EXPORT)

execute_process(
    COMMAND echo hello
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE result_var
    OUTPUT_VARIABLE output_var
    ERROR_VARIABLE error_var
)
message(STATUS result_var: ${result_var})
message(STATUS output_var: ${output_var})
message(STATUS error_var: ${error_var})

add_custom_command(
    OUTPUT hello
    COMMAND echo hello
)
add_custom_target(
    world ALL
    COMMAND echo world
    DEPENDS hello
)

# include(ExternalProject)

########################################################################################################################
# compiler and platform settings
########################################################################################################################
if (CMAKE_C_COMPILER_LOADED)
    if (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_C_FLAGS "/EHsc /Zc:__STDC__ /source-charset:utf-8")
    elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_C_FLAGS "-fexec-charset=gbk")
    endif ()
endif ()
if (CMAKE_CXX_COMPILER_LOADED)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "/EHsc /Zc:__cplusplus /source-charset:utf-8")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "-fexec-charset=gbk")
    endif ()
endif ()

set(ROOT ${PROJECT_SOURCE_DIR})

# config.h settings
include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
configure_file(
    ${PROJECT_SOURCE_DIR}/config.h.cmake
    ${PROJECT_SOURCE_DIR}/config.h
)

find_package(fmt CONFIG REQUIRED)
#target_link_libraries(main PRIVATE fmt::fmt)
#target_link_libraries(main PRIVATE fmt::fmt-header-only)
find_package(nlohmann_json CONFIG REQUIRED)
#set(nlohmann-json_IMPLICIT_CONVERSIONS OFF)
find_package(tinyxml2 CONFIG REQUIRED)
find_package(toml11 CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
#find_package(absl CONFIG REQUIRED)
#target_link_libraries(main PRIVATE absl::any absl::log absl::base absl::bits)

add_subdirectory(cc)
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

#add_subdirectory(STL)
# add_subdirectory(Windows)
# add_subdirectory(Sokyoei)
# add_subdirectory(Nono)

#set(GTEST_ROOT D:\\Download\\googletest-1.14.0\\build\\install)
find_package(GTest REQUIRED)
if (GTest_FOUND)
    include_directories(${GTEST_ROOT}/include)
    add_subdirectory(tests)
endif ()
