add_executable(any any.cpp)
add_executable(containers containers.cpp)
target_link_libraries(containers PRIVATE $<$<BOOL:${CECEILIA_USE_BOOST}>:Boost::array>)
target_link_libraries(containers PRIVATE $<$<BOOL:${CECEILIA_USE_ABSL}>:absl::any absl::log absl::base absl::bits>)
target_link_libraries(containers PRIVATE $<$<BOOL:${CECEILIA_USE_FOLLY}>:Folly::folly>)
add_executable(optional optional.cpp)

if("${CMAKE_CXX_STANDARD}" GREATER_EQUAL 20)
    add_executable(span span.cpp)
endif("${CMAKE_CXX_STANDARD}" GREATER_EQUAL 20)

add_executable(tuple tuple.cpp)
