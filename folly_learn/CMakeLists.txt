add_executable(folly_string folly_string.cpp)
target_link_libraries(folly_string PRIVATE Folly::folly Folly::folly_deps Folly::follybenchmark Folly::folly_test_util)
