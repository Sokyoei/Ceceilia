cmake_minimum_required(VERSION 3.28)  # CMake 3.28+ support C++ Modules

project(Ceceilia VERSION 0.0.1 LANGUAGES C CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)

# options
option(asio_learn "build asio_learn subproject" OFF)
option(protobuf_learn "build protobuf_learn subproject" OFF)
option(boost_learn "build boost_learn subproject" OFF)
option(folly_learn "build folly_learn subproject" OFF)

add_compile_definitions(AHRI_EXPORT)
# add_definitions(-DAHRI_EXPORT)

include(${PROJECT_SOURCE_DIR}/cmake/compiler.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/library.cmake)

set(ROOT ${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

enable_testing()
include(GoogleTest)
find_package(GTest CONFIG REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
# target_link_libraries(main PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
# add_test(AllTestsInMain main)
if(GTest_FOUND)
    set(USE_GTEST true)
endif(GTest_FOUND)

add_subdirectory(cc)
add_subdirectory(benchmark)

if(asio_learn)
    if(Boost_asio_FOUND)
        add_subdirectory(asio_learn)
    endif(Boost_asio_FOUND)
endif(asio_learn)

if(protobuf_learn)
    if(Protobuf_FOUND)
        add_subdirectory(protobuf_learn)
    endif(Protobuf_FOUND)
endif(protobuf_learn)

if(boost_learn)
    if(Boost_FOUND)
        add_subdirectory(boost_learn)
    endif(Boost_FOUND)
endif(boost_learn)

if(folly_learn)
    if(folly_FOUND)
        add_subdirectory(folly_learn)
    endif(folly_FOUND)
endif(folly_learn)

if(CMAKE_CUDA_COMPILER)
    add_subdirectory(cuda)
endif(CMAKE_CUDA_COMPILER)

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
