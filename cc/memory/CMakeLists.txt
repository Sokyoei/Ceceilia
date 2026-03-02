add_executable(jemalloc_learn jemalloc_learn.cpp)
target_link_libraries(jemalloc_learn PkgConfig::jemalloc)
add_executable(memory memory.cpp)
