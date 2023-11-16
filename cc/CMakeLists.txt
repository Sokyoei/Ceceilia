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

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_package(Threads REQUIRED)
    if(Threads_FOUND)
        add_executable(pthread concurrency/pthread.c)
        target_link_libraries(pthread PUBLIC Threads::Threads)
    endif()
endif()

# file
add_executable(xml_file file/xml_file.cpp)
target_link_libraries(xml_file PRIVATE tinyxml2::tinyxml2)
add_executable(json_file file/json_file.cpp)
target_link_libraries(json_file PRIVATE nlohmann_json::nlohmann_json)
