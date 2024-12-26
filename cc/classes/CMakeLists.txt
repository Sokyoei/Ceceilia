add_executable(test_Pimpl test_Pimpl.cpp Pimpl.hpp Pimpl.cpp)
# target_link_libraries(test_Pimpl ${GTEST_BOTH_LIBRARIES})
# add_test(NAME test_Pimpl COMMAND test_Pimpl)
target_link_libraries(test_Pimpl PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
gtest_discover_tests(test_Pimpl)
add_test(test_Pimpl test_Pimpl)

add_executable(classes classes.cpp)
add_executable(CRTP CRTP.cpp)
