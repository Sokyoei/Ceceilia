add_executable(benchmark_learn benchmark_learn.cpp)
target_link_libraries(benchmark_learn PRIVATE benchmark::benchmark benchmark::benchmark_main)
