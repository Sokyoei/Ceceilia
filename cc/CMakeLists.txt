add_executable(xml_file file/xml_file.cpp)
target_link_libraries(xml_file PRIVATE tinyxml2::tinyxml2)

add_executable(function_call function_call.cpp)

add_executable(variable_parameter variable_parameter.cpp)

find_package(OpenMP REQUIRED)
add_executable(OpenMP OpenMP.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(OpenMP PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(VLA VLA.c)
