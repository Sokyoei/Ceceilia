add_executable(any any.cpp)
add_executable(containers containers.cpp)
target_link_libraries(containers PRIVATE Boost::array)
target_link_libraries(containers PRIVATE absl::any absl::log absl::base absl::bits)
target_link_libraries(containers PRIVATE Folly::folly Folly::folly_deps Folly::follybenchmark Folly::folly_test_util)
add_executable(optional optional.cpp)

if("${CMAKE_CXX_STANDARD}" GREATER_EQUAL 20)
    add_executable(span span.cpp)
endif("${CMAKE_CXX_STANDARD}" GREATER_EQUAL 20)

add_executable(tuple tuple.cpp)
