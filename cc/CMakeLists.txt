add_executable(exceptions exceptions.cpp)
add_executable(function_call function_call.cpp)
add_executable(function_pointer function_pointer.cpp)

find_package(OpenMP REQUIRED)
add_executable(OpenMP OpenMP.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(OpenMP PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(timer timer.cpp)
add_executable(variable_parameter variable_parameter.cpp)
add_executable(VLA VLA.c)


# concurrency
add_executable(thread concurrency/thread.cpp)
add_executable(future concurrency/future.cpp)
add_executable(threadpool_example concurrency/threadpool.hpp concurrency/threadpool_example.cpp)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_package(Threads REQUIRED)
    if(Threads_FOUND)
        add_executable(pthread concurrency/pthread.c)
        target_link_libraries(pthread PUBLIC Threads::Threads)
    endif()
endif()


# file
add_executable(json_file file/json_file.cpp)
target_link_libraries(json_file PRIVATE nlohmann_json::nlohmann_json)
add_executable(toml_file file/toml_file.cpp)
target_link_libraries(toml_file PRIVATE toml11::toml11)
add_executable(xml_file file/xml_file.cpp)
target_link_libraries(xml_file PRIVATE tinyxml2::tinyxml2)
add_executable(yaml_file file/yaml_file.cpp)
target_link_libraries(yaml_file PRIVATE yaml-cpp::yaml-cpp)

# SIMD
add_executable(simd SIMD/simd.c)
if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(simd PRIVATE /arch:AVX2)
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(simd PRIVATE -mavx2)
endif()

# net
add_executable(client_example net/client.hpp net/client_example.cpp)
add_executable(server_example net/server.hpp net/server_example.cpp)
