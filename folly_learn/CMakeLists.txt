add_executable(folly_executor folly_executor.cpp)
target_link_libraries(folly_executor PRIVATE Folly::folly Folly::folly_deps Folly::follybenchmark Folly::folly_test_util)

add_executable(folly_string folly_string.cpp)
target_link_libraries(folly_string PRIVATE Folly::folly Folly::folly_deps Folly::follybenchmark Folly::folly_test_util)
