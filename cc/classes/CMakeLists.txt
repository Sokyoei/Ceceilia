add_executable(test_Pimpl test_Pimpl.cpp Pimpl.hpp Pimpl.cpp)
target_link_libraries(test_Pimpl ${GTEST_BOTH_LIBRARIES})
gtest_discover_tests(test_Pimpl)
add_test(NAME test_Pimpl COMMAND test_Pimpl)

add_executable(classes classes.cpp)
