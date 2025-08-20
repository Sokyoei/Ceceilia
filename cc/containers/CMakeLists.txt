add_executable(any any.cpp)
add_executable(containers containers.cpp)
add_executable(optional optional.cpp)

if("${CMAKE_CXX_STANDARD}" GREATER_EQUAL 20)
    add_executable(span span.cpp)
endif("${CMAKE_CXX_STANDARD}" GREATER_EQUAL 20)

add_executable(tuple tuple.cpp)
