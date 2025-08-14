add_executable(test_Ahri test_Ahri.cpp)
# target_link_libraries(test_Ahri ${GTEST_BOTH_LIBRARIES})
target_link_libraries(test_Ahri GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
gtest_discover_tests(test_Ahri)
add_test(test_Ahri test_Ahri)

add_subdirectory(Ahri/Ceceilia)
